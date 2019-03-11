clc, clear, close all

% Operating modes - in logical order. Training is done on single
% transmission scenario
% Test-out - Generates test collecton for the autoencoder
% Test-in - Decodes reconstructed LLRs from the autoencodermode = 'Test-out';
mode = 'Test-out';

% Global parameters
modSize = 8;
numRuns = 10000;
% Multiple transmission parameters
numReps = 2;
repRatio = 2/3; % Bits in each transmission

% Set seeds (for reproducible results) and SNR parameters
if strcmp(mode, 'Test-out')
    bitSeed = 7777;
    snrRangedB = 16:1:20;
elseif strcmp(mode, 'Test-in')
    bitSeed = 7777; % Needs to match Test-out mode
    snrRangedB = 16:1:20; % Needs to match Test-out mode
    % Import external LLR
    % Quantization types: FP, SQ_xbits, where x is number of bits per
    % dimension - needs to match Python output
    quantType = 'SQ_4bits';
    extLoad = load(sprintf('PY_%s_mod%d_snr%dto%d.mat', ...
        quantType, modSize, min(snrRangedB), max(snrRangedB)));
    % Extract LLRs
    extLLR = extLoad.rec_llrCollectQ;
else
    error('Invalid operating mode specified!');
end

% Interleaver seed - 1111 reproduces the results in the paper
interSeed = 1111;

% Fix bit RNG
rng(bitSeed);

% IEEE 802.11n HT LDPC
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

H  = zeros(size(rotmatrix)*Z);
Zh = diag(ones(1, Z), 0);

% Convert into binary matrix
for r=1:size(rotmatrix, 1)
    for c=1:size(rotmatrix, 2)
        rotidx = rotmatrix(r, c);
        if (rotidx > -1)
            Zt = circshift(Zh, [0 rotidx]);
        else
            Zt = zeros(Z);
        end
        limR = (r-1)*Z+1:r*Z;
        limC = (c-1)*Z+1:c*Z;
        H(limR, limC) = Zt;
    end
end

% Encoder/decoder objects
hEnc = comm.LDPCEncoder('ParityCheckMatrix', sparse(H));
hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H), 'DecisionMethod', 'Soft decision');

% System parameters
K = size(H, 1);
N = size(H, 2);
modOrder = 2^modSize;

% Interleaver
rng(interSeed);
P = randperm(N);
R(P) = 1:N;
% HARQ combinations
hIdx = zeros(numReps, N);
% First send two halves
N0 = N/numReps;
N1 = round((repRatio - 1/numReps) * N);
hIdx(1, 1:1:N0) = 1;
hIdx(2, (N0+1):1:N) = 1;
% Then send one random 1/6 piece from the other transmission
Q = randperm(N/numReps);
hIdx(1, N0 + Q(1:N1)) = 1;
hIdx(2, Q(N1+1:2*N1)) = 1;
hIdx = logical(hIdx);
rng(bitSeed)

% LLR quantization parameters (log-domain) for reference scenario
llrClip = 4; % Symmetric
llrBits = 4;
llrClipRange = linspace(-llrClip, llrClip, 2^llrBits);

% Auxiliary tables for fast LLR computation
bitmap = de2bi(0:(modOrder-1)).';
constellation = qammod(bitmap, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);

% Outputs
if any(strcmp(mode, {'Train', 'Test-out'}))
    llrCollectN = zeros(numel(snrRangedB), numRuns, numReps, N0+N1);
end

% Performance metrics
bitError = zeros(numel(snrRangedB), numRuns);
packetError = zeros(numel(snrRangedB), numRuns);

% Main loop
for snrIdx = 1:numel(snrRangedB)
    noisePower = 10 ^ (-snrRangedB(snrIdx)/10);
    for packetIdx = 1:numRuns
        % Random bits
        bitsRef = randi([0 1], K, 1);
        
        % Encode bits
        bitsEnc = hEnc(bitsRef);
        
        % Interleave bits
        bitsInt = bitsEnc(P);
        
        % Split in two transmissions
        llrSumBuffer = zeros(N, 1);
        for harqIdx = 1:numReps
            % Downselect bits
            b = bitsInt(hIdx(harqIdx, :));
            
            % Modulate bits
            x = qammod(b, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
            
            % Channel effects
            % AWGN
            n = 1/sqrt(2) * sqrt(noisePower) * (randn(size(x)) + 1i * randn(size(x)));
            % Rayleigh
            h = 1/sqrt(2) * (randn(size(x)) + 1i * randn(size(x)));
            y = h .* x + n;
            
            % Equalizer algorithm
            % Optimal inversion
            y_hat = y ./ h;
            % Channel power
            h_pow = abs(h) .^ 2;
            
            % Estimate/import LLRs
            % Always use ideal LLRs for second transmission
            if (harqIdx == 2) || strcmp(mode, 'Train')
                % Demodulate bits
                llrInt = llrComputeCSI(constellation, bitmap, modSize, y, h, noisePower);
                % Save in collection
                llrCollectN(snrIdx, packetIdx, harqIdx, :) = llrInt;
            elseif strcmp(mode, 'Test-out')
                % Demodulate bits
                llrInt = llrComputeCSI(constellation, bitmap, modSize, y, h, noisePower);
                % Save in collection
                llrCollectN(snrIdx, packetIdx, harqIdx, :) = llrInt;
                % Clipping and uniform quantization
                llrInt(abs(llrInt) > llrClip) = sign(llrInt(abs(llrInt) > llrClip)) * llrClip;
                llrInt = interp1(llrClipRange, llrClipRange, llrInt, 'nearest');
                % Room to add state of the art here
            elseif strcmp(mode, 'Test-in')
                % Use reconstructed LLRs
                llrInt = squeeze(extLLRs(snrIdx, packetIdx, harqIdx, :));
            end
             
            % Place LLR in sum-buffer
            llrSumBuffer(hIdx(harqIdx, :)) = llrSumBuffer(hIdx(harqIdx, :)) + llrInt;
        end
        
        % Deinterleave bits
        llrDeint = double(llrSumBuffer(R));
        
        % Estimate bits
        llrOut = hDec(llrDeint);
        
        % Determine bit/packet error
        bitsEst = (sign(-llrOut) +1) / 2;
        bitError(snrIdx, packetIdx) = sum(bitsRef ~= bitsEst);
        packetError(snrIdx, packetIdx) = ~all(bitsRef == bitsEst);
    end
end