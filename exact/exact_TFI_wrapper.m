function tTaken = exact_TFI_wrapper
hDetunes= [0];
hInters = [0.5];
hDrives = [1,0];
NMax    = 20;
method  = 'sa';
time_taken = zeros(NMax,1);
for i=1:numel(hDetunes)
    hDetune = hDetunes(i);
    hInter  = hInters(i);
    hDrive  = hDrives(i);
    N       = 2;
    tTaken  = zeros(N,1);
    fname   = strrep(sprintf('%s/results/exact_TFI_hDrive=%2.1f_hInter=%2.1f_hDetune=%2.1f',...
                pwd,hDrive,hInter,hDetune),'.','p');
    data    = cell(NMax-N + 1, 3);
    cntr    = 1;

    while N <= NMax
        tInit = cputime;
        [V,D] = exact_TFI(N,hDrive,hInter,hDetune,method);
        time_taken = cputime-tInit;
        fprintf('N=% d, time= %4.3f\n',N,time_taken)
        data(cntr,:)    = {N, V, D};
        tTaken(N) = time_taken;
        N    = N + 1;
        cntr = cntr +1;
        
    end

    save(fname,'data','-v7.3')

end

end

% computre expectation of H analytically