__author__ = 'Indrayana'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

numArms = 10
numTasks = 2000
numPlays = 5000

epsArray = [0, 0.01, 0.1]
QstarMeans = np.random.rand(numTasks, numArms)
QstarBestArm = np.argmax(QstarMeans, axis=1)
Qinit = np.random.rand(numTasks, numArms) + QstarMeans

avgRewardMat = np.zeros([len(epsArray), numPlays])
pctOptActionMat = np.zeros([len(epsArray), numPlays])

for epsIdx in range(0, len(epsArray)):
    eps  = epsArray[epsIdx]
    QstarEst = np.zeros(Qinit.shape)
    rewardSumMat = np.zeros(Qinit.shape)
    armCountMat = np.zeros(Qinit.shape)
    rewardMat = np.zeros([numTasks, numPlays])
    bestArmMat = np.zeros([numTasks, numPlays])

    for taskIdx in range(0, numTasks):
        for playIdx in range(0, numPlays):
            randNum = np.random.rand()
            if randNum <= eps:
                # explore
                armIdx = np.random.randint(0, numArms)
            else:
                armIdx = np.argmax(QstarEst[taskIdx,])

            reward = QstarMeans[taskIdx, armIdx] + np.random.randn()
            rewardMat[taskIdx, playIdx] = reward

            if armIdx == QstarBestArm[taskIdx]:
                bestArmMat[taskIdx, playIdx] = 1

            rewardSumMat[taskIdx, armIdx] = rewardSumMat[taskIdx, armIdx] + reward
            armCountMat[taskIdx, armIdx] += 1
            QstarEst[taskIdx, armIdx] = rewardSumMat[taskIdx, armIdx] / armCountMat[taskIdx, armIdx]

    rewardAvg = np.mean(rewardMat, axis=0)
    avgRewardMat[epsIdx,] = rewardAvg
    pctOptActionMat[epsIdx,] = np.mean(bestArmMat, axis=0)

fig, axs = plt.subplots(2, 1)
axs[0].plot(avgRewardMat[0,], label = r'$\epsilon$ = 0')
axs[0].plot(avgRewardMat[1,], label = r'$\epsilon$ = 0.01')
axs[0].plot(avgRewardMat[2,], label = r'$\epsilon$ = 0.1')
axs[0].legend()
axs[0].set_title('Average Reward')
axs[1].plot(pctOptActionMat[0,], label = r'$\epsilon$ = 0')
axs[1].plot(pctOptActionMat[1,], label = r'$\epsilon$ = 0.01')
axs[1].plot(pctOptActionMat[2,], label = r'$\epsilon$ = 0.1')
axs[1].legend()
axs[1].set_title('\% Optimal Action')

plt.show()
