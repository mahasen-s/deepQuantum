from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from main_fun import run_net
from exact.pythonLoadHDF5 import loadMatlabHDF5 as pyLoad

N = 2;

exactVals = pyLoad('./exact/results/exact_TFI_hDrive= 1p0_hInter= 0p0_hDetune= 0p0.mat','data')
run_net(N=N,alpha=4,learn_rate=0.001,optim='adam',M=100,mcs=1000,h_drive=1,h_inter=0.5,h_detune=0,wf_exact=exactVals[N-2],fileOut='hDrive=1_hInter=0')

exactVals = pyLoad('./exact/results/exact_TFI_hDrive= 0p0_hInter= 1p0_hDetune= 0p0.mat','data')
run_net(N=N,alpha=4,learn_rate=0.1,optim='adam',M=100,mcs=1000,h_drive=1,h_inter=0.5,h_detune=0,wf_exact=exactVals[N-2],fileOut='hDrive=0_hInter=1')
