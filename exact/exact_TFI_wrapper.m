function exact_TFI_wrapper
hDetunes= [0,0];
hInters = [0,1];
hDrives = [1,0];
NMax    = 15;
method  = 'sa';

for i=1:numel(hDetunes)
    hDetune = hDetunes(i);
    hInter  = hInters(i);
    hDrive  = hDrives(i);
    N       = 2;
    tTaken  = 0;
    fname   = strrep(sprintf('%s/results/exact_TFI_hDrive=%4.1f_hInter=%4.1f_hDetune=%4.1f',...
                pwd,hDrive,hInter,hDetune),'.','p');
    data    = cell(NMax-N + 1, 3);
    cntr    = 1;

    while N <= NMax
        tInit = cputime;
        [V,D] = exact_TFI(N,hDrive,hInter,hDetune,method);
        fprintf('N=%d, time= %4.3f\n',N,cputime-tInit)
        data(cntr,:)    = {N, V, D};
        N    = N + 1;
        cntr = cntr +1;
    end

    save(fname,'data')

end

end
