% Inputs: Code type, Channel type, Number of packets, SNR Range, Seed
% Code type = {'ldpc', 'polar'}
% Channel type = {'rayleigh', 'fading'}
% Outputs: Reference BLER/BER
% Byproducts: Writes reference bit files and LLR files in 'target_folder'
function [bler, ber] = GenTrainingData(channel_type, code_type, num_packets, ...
    snr_range, bit_seed, target_folder)

% Modulation size (currently hardcoded at 8)
mod_size  = 8;
mod_order = 2^mod_size;
% Interleaver seed - do NOT generally change
inter_seed = 1111;

% Code parameters
if strcmp(code_type, 'ldpc')
    % Taken from IEEE 802.11n: HT LDPC matrix definitions
    % You can change this according to your needs
    Z = 27;
    rotmatrix = ...
        [0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
        2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
        23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
        24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
        25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
        13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
        7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
        11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
        25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
        3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];

    H = zeros(size(rotmatrix)*Z);
    Zh = diag(ones(1,Z),0);

    % Convert into binary matrix
    for r=1:size(rotmatrix,1)
        for c=1:size(rotmatrix,2)
            rotidx = rotmatrix(r,c);
            if (rotidx > -1)
                Zt = circshift(Zh,[0 rotidx]);
            else
                Zt = zeros(Z);
            end
            limR = (r-1)*Z+1:r*Z;
            limC = (c-1)*Z+1:c*Z;
            H(limR,limC) = Zt;
        end
    end

    hEnc = comm.LDPCEncoder('ParityCheckMatrix', sparse(H));
    hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H), 'DecisionMethod', 'Soft decision', ...
        'IterationTerminationCondition', 'Parity check satisfied', 'MaximumIterationCount', 50);

    % System parameters
    K = size(H, 1);
    N = size(H, 2);
elseif strcmp(code_type, 'polar')
    % Polar code parameters
    K = 128;
    N = 256;
    L = 4; % List length
else
    error('Invalid code type!')
end

% Channel type
if strcmp(channel_type, 'rayleigh')
    % Passthrough
elseif strcmp(channel_type, 'fading')
    % Universal dirac impulse
    dirac_in = zeros(N, 1);
    dirac_in(1) = 1;
else
    error('Invalid channel type!')
end

% Interleaver
rng(inter_seed);
P = randperm(N);
R(P) = 1:N;
rng(bit_seed)

% Auxiliary tables for fast LLR computation
bitmap = de2bi(0:(mod_order-1)).';
constellation = qammod(bitmap, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);

% Performance metrics
ber  = zeros(numel(snr_range), num_packets);
bler = zeros(numel(snr_range), num_packets);

% Byproducts
ref_bits = zeros(numel(snr_range), num_packets, K);
ref_llr = zeros(numel(snr_range), num_packets, N);

% Progress
progressbar(0, 0);

% Main loop
for snr_idx = 1:numel(snr_range)
    noise_power = 10 ^ (-snr_range(snr_idx)/10);
    for run_idx = 1:num_packets
        % Random bits
        payload_bits = randi([0 1], K, 1);
        % Save in collection
        ref_bits(snr_idx, run_idx, :) = payload_bits;
        
        % Encode bits
        if strcmp(code_type, 'ldpc')
            bitsEnc = hEnc(payload_bits);
        elseif strcmp(code_type, 'polar')
            bitsEnc = nrPolarEncode(payload_bits, N);
        end
        
        % Interleave bits
        bitsInt = bitsEnc(P);
        
        % Modulate bits
        x = qammod(bitsInt, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);
        
        % Channel effects
        % AWGN
        n = 1/sqrt(2) * sqrt(noise_power) * (randn(N/mod_size, 1) + 1i * randn(N/mod_size, 1));
        if strcmp(channel_type, 'rayleigh')
            % Draw random N(0, 1)
            h = 1/sqrt(2) * (randn(size(x)) + 1i * randn(size(x)));
        elseif strcmp(channel_type, 'fading')
            % Fading channel
            % Reset seed every packet
            channel.Seed = inter_seed+snr_idx+run_idx;
            channel.NRxAnts = 1;
            channel.DelayProfile = 'ETU';
            channel.DopplerFreq = 0;
            channel.CarrierFreq = 2e9;
            channel.MIMOCorrelation = 'Low';
            channel.SamplingRate = 1/1e-7; % 10 MHz
            channel.InitTime = 0;        
            % Derive impulse response
            hImp  = lteFadingChannel(channel, dirac_in);
            hFreq = fft(hImp);
            % Normalize to unit average power
            hFreq = hFreq / sqrt(mean(abs(hFreq) .^ 2));
            % Scramble subcarriers
            rng(inter_seed+1)
            hFreq = hFreq(randperm(N));

            % Downsample channel
            h = hFreq(1:mod_size:end);
        end
        
        % Apply channel
        y = h .* x + n;
        
        % LLR computation
        llrInt = ComputeLLR(constellation, bitmap, mod_size, y, h, noise_power);
        % Save in collection
        ref_llr(snr_idx, run_idx, :) = llrInt;
        
        % Deinterleave bits
        llrDeint = double(llrInt(R));
        
        % Channel decoder
        if strcmp(code_type, 'ldpc')
            llrOut = hDec(llrDeint);
            bitsEst = (sign(-llrOut) +1) / 2;
        elseif strcmp(code_type, 'polar')
            bitsEst = nrPolarDecode(llrDeint, K, N, L);
        end
        
        % Determine bit/packet error
        ber(snr_idx, run_idx)  = mean(payload_bits ~= bitsEst);
        bler(snr_idx, run_idx) = ~all(payload_bits == bitsEst);

        % Progress
        progressbar(double(run_idx) / double(num_packets), []);
    end
    % Progress
    progressbar([], snr_idx / numel(snr_range));
end
progressbar(1, 1)

% Average results
ber = mean(ber, 2);
bler = mean(bler, 2);

% Save byproduct bits
filename = sprintf('%s/ref_bits_mod%d_seed%d.mat', ...
    target_folder, mod_size, bit_seed);
save(filename, 'ref_bits', '-v7.3');
% Save byproduct LLR
filename = sprintf('%s/ref_llr_mod%d_seed%d.mat', ...
    target_folder, mod_size, bit_seed);
save(filename, 'ref_llr', '-v7.3');

end