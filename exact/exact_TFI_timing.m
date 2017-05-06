function exact_TFI_timing
hList   = 0.5;%:0.25:2.0;
NMax    = 15;
method  = 'sa';

for i=1:numel(hList)
    h       = hList(i);
    N       = 2;
    tTaken  = 0;
    fname   = strrep(sprintf('%s/results/exact_TFI_h=%4.3f',pwd,h),'.','p');
    data    = cell(NMax-N + 1, 3);
    cntr    = 1;

    while N <= NMax
        tInit = cputime;
        [V,D] = exact_TFI(N,h,1,0,method);
        fprintf('N=%d, time= %4.3f\n',N,cputime-tInit)
        data(cntr,:)    = {N, V, D};
        N    = N + 1;
        cntr = cntr +1;
    end

    save(fname,'data')

end

end
