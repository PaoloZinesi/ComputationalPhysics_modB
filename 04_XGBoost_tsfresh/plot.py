import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import non_negative_factorization

# plot of the first 3 time series
def Show_data(x,L, s="data", nseries=3, A=None, axis=None, show=True, dolegend=True): 
    if axis is None:
        _, axis = plt.subplots()
        
    for i in range(nseries):
        if A is None: lab = 'Time series '+str(i)
        else: lab = 'Time series '+str(i)+' - A='+str(A)
        axis.plot(np.arange(0+i*L, L*(i+1)), x[i], label=lab)
    
    axis.set_title(s)
    axis.minorticks_on()
    axis.set_xlabel("time")
    if dolegend: 
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
        shadow=True, framealpha=1, facecolor='aliceblue', edgecolor='black', prop={'size':8})

    plt.tight_layout()
    if show: plt.show()
    

# def Show_data_scatter(x,L,s="data", nseries=3):
#     for i in range(nseries):
#         plt.scatter(np.arange(0+i*L, L*(i+1)), x[i], label='Time series '+str(i), s=4)
# 
#     plt.title(s)
#     plt.xlabel("time")
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
#         shadow=True, framealpha=1, facecolor='aliceblue', edgecolor='black', prop={'size':8})
# 
#     if show: plt.show()

def Show_weights(model,l=0,label="model", show=True):
    c=['r','y','c','b','m', 'g']
    m=['o','s','D','<','>', '*']
    ms=10
    
    w = model.layers[l].get_weights()[0]#weights
    wT=w.T
    M=len(wT)
    b = model.layers[l].get_weights()[1]#bias
    fig,AX=plt.subplots(1,2,figsize=(12,5))
    ax=AX[0]
    ax.axhline(0, c="k")
    ax.plot((0,))
    for i in range(M):
        ax.plot(wT[i][0],"-",c=c[i],marker=m[i],label=str(i),markersize=ms)
    ax.set_title(label+': filters of layer '+str(l))
    ax.set_xlabel('index')
    ax=AX[1]
    ax.axhline(0, c="k")
    for i in range(M):
        ax.plot((i),(b[i]),c=c[i],marker=m[i],label="filter "+str(i),markersize=ms)
    ax.set_title(label+': bias of layer '+str(l))
    ax.set_xlabel('filter nr')
    ax.set_xticks(np.arange(5))
    ax.legend()
    if show: plt.show()

def Show_history(fit, EPOCHS=100, show=True):
    fig,AX=plt.subplots(1,2,figsize=(12,5.))
    ax=AX[0]
    ax.plot(fit.history['accuracy'],"b",label="train")
    ax.plot(fit.history['val_accuracy'],"r--",label="valid.")
    ax.plot((0,EPOCHS),(1/3,1/3),":",c="gray",label="random choice")
    ax.set_xlabel('epoch')
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.legend()
    ax=AX[1]
    ax.plot(fit.history['loss'],"b",label="train")
    ax.plot(fit.history['val_loss'],"r--",label="valid.")
    ax.set_xlabel('epoch')
    ax.set_ylabel("Loss")
    ax.set_ylim([0, 1.05*np.max(fit.history['loss'])])
    ax.legend()
    if show: plt.show()