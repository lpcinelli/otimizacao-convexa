import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plotConvergence(xHist,cost,save=False,path='.'):
   
    datapoints = np.asarray(xHist)
    minVals = np.min(datapoints,axis=0)
    maxVals = np.max(datapoints,axis=0)
    minVals -= 0.5
    maxVals += 0.5
    domainX = np.linspace(minVals[0], maxVals[0], 300)
    domainY = np.linspace(minVals[1], maxVals[1], 300)
    X, Y = np.meshgrid(domainX, domainY)
    Z = cost(X, Y)

    sns.set()
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes()
    ax.contour(X, Y, Z, 100)

    for j in range(1, datapoints.shape[0]):

        ax.annotate('', xy=(datapoints[j, :]), xytext=(datapoints[j-1, :]),
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
    if save is True:
        fig.tight_layout()
        fig.savefig(path, pad_inches=0)
