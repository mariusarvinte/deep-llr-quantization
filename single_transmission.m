clc, clear, close all

% Operating modes - in logical order
% Train - Generates training collection for the autoencoder
% Test-out - Generates test collecton for the autoencoder
% Test-in - Decodes reconstructed LLRs from the autoencoder
mode = 'Train';

% Global parameters
modSize = 8;
numRuns = 10000;

% Set seeds (for reproducible results) and SNR parameters
if strcmp(mode, 'Train')
    bitSeed = 8888; % Exactly as in the paper
    snrRangedB = 16:1:20; % Python needs to match this
elseif strcmp(mode, 'Test-out')
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

H = zeros(size(rotmatrix)*Z);
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
rng(bitSeed)

% Auxiliary tables for fast LLR computation
bitmap = de2bi(0:(modOrder-1)).';
constellation = qammod(bitmap, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);

% Outputs
if any(strcmp(mode, {'Train', 'Test-out'}))
    llrCollect = zeros(numel(snrRangedB), numRuns, N);
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
        
        % Modulate bits
        x = qammod(bitsInt, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
        
        % Channel effects
        % Noise
        n = 1/sqrt(2) * sqrt(noisePower) * (randn(N/modSize, 1) + 1i * randn(N/modSize, 1));
        % Rayleigh
        h = 1/sqrt(2) * (randn(size(x)) + 1i * randn(size(x)));
        y = h .* x + n;
          
        % Estimate/import LLRs
        if any(strcmp(mode, {'Train', 'Test-out'}))
            % Demodulate bits
            % Estimate LLR (closed-form)
            llrInt = compute_llr(constellation, bitmap, modSize, y, h, noisePower);
            % Save in collection
            llrCollect(snrIdx, packetIdx, :) = llrInt;
        elseif strcmp(mode, 'Test-in')            
            % Use reconstructed LLRs
            llrInt = squeeze(extLLR(snrIdx, packetIdx, :));
        end
        
        % Deinterleave bits
        llrDeint = double(llrInt(R));
        
        % Estimate LLR
        llrOut = hDec(llrDeint);
        
        % Determine bit/packet error
        bitsEst = (sign(-llrOut) +1) / 2;
        bitError(snrIdx, packetIdx) = sum(bitsRef ~= bitsEst);
        packetError(snrIdx, packetIdx) = ~all(bitsRef == bitsEst);
    end
end

% Save LLR collection
if any(strcmp(mode, {'Train', 'Test-out'}))
    fileName = sprintf('MAT_Rayleigh_%s_mod%d_snr%dto%d.mat', ...
        mode, modSize, min(snrRangedB), max(snrRangedB));
    save(fileName, 'llrCollect');
end