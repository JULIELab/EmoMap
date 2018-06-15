import pandas as pd 
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
matplotlib.use('Agg') #change backend for headless operation
import matplotlib.pyplot as plt 
from framework import util
import numpy as np
from main.data import VAD, BE5

df=util.load_tsv('overview.tsv')


ind=np.arange(1, 9)
print(ind)




fig, ax=plt.subplots()

ax.bar(ind[:3], df.loc['Average',VAD], 
	color='blue', edgecolor='black', linewidth=1, hatch='/',
	zorder=3, label='VAD') #zorder determines foreground-background ordering

ax.bar(ind[3:], df.loc['Average',BE5], 
	color='red', edgecolor='black', linewidth=1, hatch='\\',
	zorder=3, label='BE5') #zorder determines foreground-background ordering

plt.xticks(ind, list(df), rotation=45)

ax.grid(zorder=0)
ax.xaxis.grid(False) #turns vertical grid lines off
ax.legend(fontsize=15)

plt.tight_layout()
fig.savefig('plot.pdf')