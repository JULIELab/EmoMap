import pandas as pd 
import numpy as np 
from framework.util import save_tsv, no_zeros_formatter
from main.data import SETTINGS, IN_PAPER_NAMES, VAD, BE5, VA, SHORT_COLUMNS
import datetime

df=pd.read_excel('../shr.xlsx', sheet_name=None, index_col=0)

# print(df)

def undo_spearman_brown(x):
	'''
	Reverts Spearman-Brown adjustment for k=2 (the case of adjusting
	split-half reliability).
	'''
	return 1/((2/x)-(2-1))

def spearman_brown_adjustment(r,k):
	r_adjusted=(k*r)/(1+(k-1)*r)
	return r_adjusted

save_tsv(df.drop('CORRECTED', axis=1), 'shr_as_reported.tsv')

# print(df)

'''
Normalize split-half reliabilities to N=10. That is, apply spearman_brown_
adjustment with k=10/N if the score has already been normalized (e.g., N=30
then each half contains ratings from 15 participants. Then the predicted reliability
for N=30 was reported). Otherwise, if the reported split-half-reliabilities has
not already been adjusted, k is set to 10/(N/2). E.g., N=20, then 10 ratings are 
in each split, so k=10/(20/2)=1, so the reported reliabilities remain unchanged.
'''
N_star=10#10
for i in range(df.shape[0]):
	if df['CORRECTED'][i]==1:
		# for j in range(df.shape[1]):
		# 	df.iloc[i,j]=undo_spearman_brown(df.iloc[i,j])
		for j in range(df.shape[1]):
			df.iloc[i,j]=spearman_brown_adjustment(r=df.iloc[i,j], k=N_star/df['N'][i])
	elif df['CORRECTED'][i]==0:
		for j in range(df.shape[1]):
			df.iloc[i,j]=spearman_brown_adjustment(r=df.iloc[i,j], k=N_star/(df['N'][i]/2))

# print(df)
#save_tsv(df.drop('CORRECTED', axis=1), 'shr_normalized.tsv')

### arrange shr into table

df_shr=pd.DataFrame(index=[s.name for s in SETTINGS], columns=VAD+BE5)
df_shr.rename(index=IN_PAPER_NAMES, inplace=True)
df_shr.loc['en_2', VAD]=df.loc['Warriner et al 13', VAD]
df_shr.loc['es_1', BE5]=df.loc['Ferre et al 16', BE5]
df_shr.loc['es_2', VAD+BE5]=df.loc['Hinojosa et al., 16', VAD+BE5]
df_shr.loc['es_3', VA]=df.loc['Stadthagen et al 2016', VA]
df_shr.loc['es_3', BE5]=df.loc['Stadthagen et al 2017', BE5]
df_shr.loc['pl_1',VA]=df.loc['Riegel et al., 15',VA]
df_shr.loc['pl_1', BE5]=df.loc['Wierzba et al.', BE5]
df_shr.loc['pl_2', VAD]=df.loc['Imbir et al., 17', VAD]
df_shr.loc['pl_2', BE5]=df.loc['Wierzba et al.', BE5]
df_shr.rename(index=str, columns=SHORT_COLUMNS, inplace=True)

save_tsv(df_shr,'shr_normalized.tsv')
string=df_shr.to_latex(float_format=no_zeros_formatter, na_rep='---')
string=string.replace('\\toprule','\hline')
string=string.replace('\midrule','\hline\hline')
string=string.replace('\\bottomrule','\hline')
lines=string.split('\n')
lines[0]='\\begin{tabular}{|l|rrr|rrrrr|}'

lines.insert(0,'%%%%%% Automatic Python output from {} &%%%%%%%%%%'.format(datetime.datetime.now()))
lines.insert(-1, '%%%%%%%%%%%%%%%%%%%%%%%%')
string='\n'.join(lines)
print(string)
print(string)

