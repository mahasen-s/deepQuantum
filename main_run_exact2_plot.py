import matplotlib.pyplot as plt
import numpy as np

dataDrive   = np.load('./hDrive=1_hInter=0_ss.npz')
dataInter   = np.load('./hDrive=0_hInter=1_ssi.npz')
mcs         = 200;

plt.figure(1)

plt.subplot(221)
plt.plot(np.arange(mcs),dataDrive['E_var_list'])
plt.xlabel('Epoch')
plt.ylabel('E_Var')
plt.title('hDrive=1, hInter=0')
plt.grid(True)

plt.subplot(223)
plt.plot(np.arange(mcs),dataDrive['Overlap_list'],marker='o')
plt.xlabel('Epoch')
plt.ylabel('Overlap')
plt.title('hDrive=1, hInter=0')
plt.grid(True)

plt.subplot(222)
plt.plot(np.arange(mcs),dataInter['E_var_list'])
plt.xlabel('Epoch')
plt.ylabel('E_var')
plt.title('hDrive=0, hInter=1')
plt.grid(True)

plt.subplot(224)
plt.plot(np.arange(mcs),dataInter['Overlap_list'],marker='o')
plt.xlabel('Epoch')
plt.ylabel('Overlap')
plt.title('hDrive=1, hInter=0')
plt.grid(True)
plt.show()
