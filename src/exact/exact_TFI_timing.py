from timeit import default_timer as timer
from exact_TFI import exact_TFI

h       = 0.5
Nmax    = 8
tTaken  = 0
N       = 2

while tTaken < 1000:
    startTime       = timer()
    evals, evecs    = exact_TFI(N,h)
    endTime         = timer()
    tTaken          = endTime -startTime;
    print('N=%d, time= %4.3fs\n' % (N,tTaken))
    N += 1
