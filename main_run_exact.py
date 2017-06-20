from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from main_fun import run_net
from exact.pythonLoadHDF5 import loadMatlabHDF5 as pyLoad

N = 2;
exactVals = pyLoad('./exact/results/exact_TFI_h=0p500.mat','data')
run_net(N=N,alpha=4,learn_rate=0.1,optim='adam',M=1000,mcs=1000,h_drive=1,h_inter=0.5,h_detune=0,wf_exact=exactVals[N-2])
