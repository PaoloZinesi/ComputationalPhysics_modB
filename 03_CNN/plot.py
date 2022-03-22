import numpy as np
import matplotlib.pyplot as plt

# plot of the first 3 time series
def Show_data(x,L,s="data", nseries=3, show=True):
    for i in range(nseries):
        plt.plot(np.arange(0+i*L, L*(i+1)), x[i], label='Time series '+str(i))
    
    plt.title(s)
    plt.xlabel("time")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
        shadow=True, framealpha=1, facecolor='aliceblue', edgecolor='black', prop={'size':8})

    if show: plt.show()

def Show_data_scatter(x,L,s="data", nseries=3):
    for i in range(nseries):
        plt.scatter(np.arange(0+i*L, L*(i+1)), x[i], label='Time series '+str(i), s=4)

    plt.title(s)
    plt.xlabel("time")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
        shadow=True, framealpha=1, facecolor='aliceblue', edgecolor='black', prop={'size':8})

    if show: plt.show()