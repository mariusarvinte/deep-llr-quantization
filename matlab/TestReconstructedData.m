% Takes reference file and reconstructed LLR file as input
% Works with code type = {'ldpc', 'polar'}
function [bler, ber] = TestReconstructedData(ref_file, in_file, code_type)

% Load reference bit files
contents = load(ref_file);
ref_bits = contents.ref_bits;
% Load input llr files
contents = load(in_file);
llr_rec  = contents.rec_llr;

% Derive global parameters
num_snr     = size(ref_bits, 1);
num_packets = size(ref_bits, 2);
% Sanity checks
if size(llr_rec, 1) ~= num_snr || size(llr_rec, 2) ~= num_packets
    error('Size mismatch between ref/Python!')
end

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

% Interleaver
rng(inter_seed);
P = randperm(N);
R(P) = 1:N;

% Performance metrics
bitError = zeros(num_snr, num_packets);
packetError = zeros(num_snr, num_packets);

% Progress
progressbar(0, 0);

% Main loop
for snr_idx = 1:num_snr
    for packet_idx = 1:num_packets      
        % Estimate/import LLRs
        % Use reconstructed LLRs
        llrInt = squeeze(llr_rec(snr_idx, packet_idx, :));

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
        bitError(snr_idx, packet_idx) = sum(squeeze(ref_bits(snr_idx, packet_idx, :)) ~= bitsEst);
        packetError(snr_idx, packet_idx) = ~all(squeeze(ref_bits(snr_idx, packet_idx, :))  == bitsEst);

        % Progress
        progressbar(packet_idx / num_packets, []);
    end
    % Progress
    progressbar([], snr_idx / num_snr);
end
progressbar(1, 1)

% Compute output vallues
bler = mean(packetError, 2);
ber  = mean(bitError/K, 2);

end