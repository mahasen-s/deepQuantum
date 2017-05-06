# Example of run file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from main_fun import run_net


run_net(N=2,alpha=2,learn_rate=0.1,optim='gradient_descent',M=1000,mcs=1000,h_drive=1,h_inter=0.5,h_detune=0,wf_exact=[],fileOut=[],verbose=False)
