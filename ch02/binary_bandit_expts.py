__author__ = 'Indrayana'

import numpy as np
import matplotlib.pyplot as plt

probs_win = [0.2, 0.1]
#probs_win = [0.9, 0.8]
#probs_win = [0.7, 0.3]
#probs_win = [0.2, 0.9]

numTasks = 2000
numPlays = 500
numArms = 2 # binary

bestArm = np.argmax(probs_win)
epsArray = [0, 0.1, np.nan, np.nan]
# epsArray[2] : LR-P
# epsArray[3] : LR-I
pctOptActionMat = np.zeros([len(epsArray), numPlays])
alpha = 0.1

for epsIdx in range(0, len(epsArray)):
    eps = epsArray[epsIdx]
    bestArmMat = np.zeros([numTasks, numPlays])

    for taskIdx in range(0, numTasks):
        armIdx = np.random.randint(0, numArms)
        probsSelect = np.array([0.5, 0.5])
        for playIdx in range(0, numPlays):
            randNum = np.random.rand()
            if not(np.isnan(eps)) and randNum <= eps:
                armIdx = np.random.randint(0, numArms)
            elif epsIdx >= 2:
                armIdx = np.random.choice(np.arange(0, numArms), p=probsSelect)

            if armIdx == bestArm:
                bestArmMat[taskIdx, playIdx] = 1

            updateProb = False
            if epsIdx >= 2:
                updateProb = True

            randNum2 = np.random.rand()
            if randNum2 > probs_win[armIdx]:
                # switch arm
                armIdx = np.mod(armIdx+1, 2)

                if epsIdx == 3:
                    updateProb = False

            if updateProb:
                otherArmIdx = np.mod(armIdx+1, 2)
                probsSelect[armIdx] += alpha * (1.0 - probsSelect[armIdx])
                probsSelect[otherArmIdx] = 1.0 - probsSelect[armIdx]

    pctOptActionMat[epsIdx,] = np.mean(bestArmMat, axis=0)

fig, axs = plt.subplots(1, 1)
axs.plot(pctOptActionMat[0,], label = r'$\epsilon$ = 0')
axs.plot(pctOptActionMat[1,], label = r'$\epsilon$ = 0.1')
axs.plot(pctOptActionMat[2,], label = r'$L_{R-P}$')
axs.plot(pctOptActionMat[3,], label = r'$L_{R-I}$')
axs.legend()
axs.set_title('Average Reward')

plt.show()