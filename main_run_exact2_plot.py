import matplotlib.pyplot as plt
import numpy as np

dataDrive   = np.load('./hDrive=1_hInter=0.npz')
dataInter   = np.load('./hDrive=0_hInter=1.npz')
mcs         = 1000;

plt.figure(1)

plt.subplot(121)
plt.plot(np.arange(mcs),dataDrive['E_var_list'])
plt.xlabel('Epoch')
plt.ylabel('E_Var')
plt.title('hDrive=1, hInter=0')
plt.grid(True)

plt.subplot(122)
plt.plot(np.arange(mcs),dataInter['E_var_list'])
plt.xlabel('Epoch')
plt.ylabel('E_Var')
plt.title('hDrive=0, hInter=1')
plt.grid(True)

plt.show()
