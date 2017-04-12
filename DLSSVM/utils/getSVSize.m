function nSVs = getSVSize(patterns)
% Return the number of all support vectors

n = size(patterns, 2);
nSVs = 0;
for i = 1:n
    nSVs = nSVs + size(patterns{i}.supportVectorNum,2);
end