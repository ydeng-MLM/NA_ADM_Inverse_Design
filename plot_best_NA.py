import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

datadir = "D:/Duke/MM_MM_Project/14_parameter/NA/meta_material/1milion_sweep"
fname = "30k_extrah"

print("Start reading file...")
data_x = pd.read_csv(str(datadir)+'/test_Xpred_point'+str(fname)+'inference0.csv', delimiter=' ', header=None)
print("Finished x pred!")
data_truth = pd.read_csv(str(datadir)+'/test_Ytruth_'+str(fname)+'.csv', delimiter=' ', header=None)
print("Finished y truth!")
data_pred = pd.read_csv(str(datadir)+'/test_Ypred_point'+str(fname)+'inference0.csv', delimiter=' ', header=None)
print("Finished data loading!")

xdata = data_x.to_numpy()
pred = data_pred.to_numpy()
truth = data_truth.to_numpy()

geoboundary = [0.3, 0.75, 1, 1.5, 0.1, 0.2, -0.7854, 0.7854]

truth = np.array([truth[0] for i in range(len(pred))])

print(len(xdata), len(pred), len(truth))

mse = np.mean(np.square(pred[:, :875] - truth[:, :875]), axis=1)
i = np.argmin(mse)
print(i)
'''
freq = np.linspace(100, 500, 1900)
plt.rc('font', family='serif', size='15')

fig, ax1 = plt.subplots()

ax1.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
ax1.set_xlabel('Frequency (THz)')
ax1.set_ylabel('Absorptance')
a, = ax1.plot(freq, truth[i][50:-50], '#952500')
b, = ax1.plot(freq, pred[i][50:-50], color='#00a7c9')
#c, = ax1.plot(freq, A[50:-50], color='k')
ax1.set_ylim([0, 1])
ax1.set_xlim([100, 500])
ax1.text(110, 0.9, "(a)")
ax1.text(230, 0.9, "MSE={:.2e}".format(mse[i]))

#ax1.legend(com, ['GaSb EQE', 'DNN Prediction'], loc='lower right', fontsize=15)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.set_size_inches(6, 4)

fig.savefig('D:/Duke/MM_MM_Project/14_parameter/NA/meta_material/1m_test.jpg',dpi=300)
'''


mse_in = np.array([np.min(mse[:100000]), np.min(mse[:200000]), np.min(mse[:300000]), np.min(mse[:400000]), np.min(mse[:500000]), np.min(mse[:600000]), np.min(mse[:700000]), np.min(mse[:800000]), np.min(mse[:900000]), np.min(mse)])
num_in = np.linspace(1, 10, 10)
plt.rc('font', family='serif', size='15')

fig, ax1 = plt.subplots()

ax1.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
ax1.set_xlabel('# of initialization')
ax1.set_ylabel('MSE')
a, = ax1.plot(num_in, mse_in, '#952500')


#ax1.legend(com, ['GaSb EQE', 'DNN Prediction'], loc='lower right', fontsize=15)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.set_size_inches(6, 4)

fig.savefig('D:/Duke/MM_MM_Project/14_parameter/NA/meta_material/1m_test.jpg',dpi=300)