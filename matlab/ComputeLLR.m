% Auxiliary function that computes full precision, exact LLR values with
% CSI knowledge
% Bitmap needs to match the symbols in 'constellation'
function llrOut = ComputeLLR(constellation, bitmap, modSize, y, h, sigmaN)

% Initialize
llrOut = zeros(modSize * numel(y), 1);

% For each bit, compute zero set / one set
for bitIdx = 1:modSize    
    zeroSymbols = constellation(bitmap(bitIdx, :) == 0);
    oneSymbols = constellation(bitmap(bitIdx, :) == 1);

    zeroDiffs = sum(exp(-abs(y - h .* zeroSymbols) .^ 2 / (2*sigmaN)), 2);
    oneDiffs  = sum(exp(-abs(y - h .* oneSymbols) .^ 2 / (2*sigmaN)), 2);
    
    llrLocal = log(zeroDiffs ./ oneDiffs);
    
    llrOut(bitIdx:modSize:end) = llrLocal;
end