__author__ = 'Indrayana'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

numArms = 10
numTasks = 2000
numPlays = 1000

tempArray = [0.01, 0.1, 1]
QstarMeans = np.random.rand(numTasks, numArms)
QstarBestArm = np.argmax(QstarMeans, axis=1)
Qinit = np.random.rand(numTasks, numArms) + QstarMeans

avgRewardMat = np.zeros([len(tempArray), numPlays])
pctOptActionMat = np.zeros([len(tempArray), numPlays])

for tempIdx in range(0, len(tempArray)):
    temp  = tempArray[tempIdx]
    QstarEst = np.zeros(Qinit.shape)
    rewardSumMat = np.zeros(Qinit.shape)
    armCountMat = np.zeros(Qinit.shape)
    rewardMat = np.zeros([numTasks, numPlays])
    bestArmMat = np.zeros([numTasks, numPlays])

    for taskIdx in range(0, numTasks):
        for playIdx in range(0, numPlays):
            QstarEst_exp = np.exp(QstarEst[taskIdx,] / temp)
            prob = QstarEst_exp / np.sum(QstarEst_exp)
            armIdx = np.random.choice(np.arange(numArms), p=prob)

            reward = QstarMeans[taskIdx, armIdx] + np.random.randn()
            rewardMat[taskIdx, playIdx] = reward

            if armIdx == QstarBestArm[taskIdx]:
                bestArmMat[taskIdx, playIdx] = 1

            rewardSumMat[taskIdx, armIdx] = rewardSumMat[taskIdx, armIdx] + reward
            armCountMat[taskIdx, armIdx] += 1
            QstarEst[taskIdx, armIdx] = rewardSumMat[taskIdx, armIdx] / armCountMat[taskIdx, armIdx]

    rewardAvg = np.mean(rewardMat, axis=0)
    avgRewardMat[tempIdx,] = rewardAvg
    pctOptActionMat[tempIdx,] = np.mean(bestArmMat, axis=0)

fig, axs = plt.subplots(2, 1)
axs[0].plot(avgRewardMat[0,], label = r'$\tau$ = 0.01')
axs[0].plot(avgRewardMat[1,], label = r'$\tau$ = 0.1')
axs[0].plot(avgRewardMat[2,], label = r'$\tau$ = 1')
axs[0].legend()
axs[0].set_title('Average Reward')
axs[1].plot(pctOptActionMat[0,], label = r'$\tau$ = 0.01')
axs[1].plot(pctOptActionMat[1,], label = r'$\tau$ = 0.1')
axs[1].plot(pctOptActionMat[2,], label = r'$\tau$ = 1')
axs[1].legend()
axs[1].set_title('\% Optimal Action')

plt.show()
