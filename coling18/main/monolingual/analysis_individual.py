import numpy as np
import pandas as pd 
from main.data import SETTINGS, IN_PAPER_NAMES, VAD, BE5, SHORT_COLUMNS
from framework.util import get_average_result_from_df, save_tsv, no_zeros_formatter, load_tsv
import datetime
import framework.util as util

directions=['be2vad', 'vad2be']

models=['baseline', 'reference_LM', 'Reference_KNN', 'my_model']
VARS=VAD+BE5

df=pd.DataFrame(index=[setting.name for setting in SETTINGS],
				columns=VARS)


for d in directions:
	for s in SETTINGS:
		results=load_tsv('results/{}/{}/my_model.tsv'.format(d, s.name))
		for var in VARS:
			if var in list(results):
				df.loc[s.name, var]=results.loc['Average', var]


df.rename(index=IN_PAPER_NAMES, inplace=True)
df.rename(index=str, columns=SHORT_COLUMNS, inplace=True)
save_tsv(df, 'overview_individual.tsv')

# read normalized split half reliabilites to make larger values bold            
df_shr=load_tsv('../../analysis/shr/shr_normalized.tsv')
df_greater=df>df_shr
df_lesser=df<df_shr
print(df_greater)
print(df_lesser)

outperformed=0
not_outperformed=0

# add cell colour
df=df.round(3)
print(df)


lines=[]
lines.append('%%%%%% Automatic Python output from {} &%%%%%%%%%%'.format(datetime.datetime.now()))
lines.append('\\begin{tabular}{|l|rrr|rrrrr|}')
lines.append('\hline')
lines.append(' & '.join(['{}']+list(df))+'\\\\')
lines.append('\hline\hline')

for i in range(df.shape[0]):
	row_list=[]
	row_list.append(df.index[i].replace('_','\_'))
	for j in range(df.shape[1]):
		cell=''
		if df_greater.iloc[i,j]==True:
			cell+='\cellcolor{blue!25} '
		elif df_lesser.iloc[i,j]==True:
			cell+='\cellcolor{lightred} '
		cell+=util.no_zeros_formatter(df.iloc[i,j])
		row_list.append(cell)
	lines.append(' & '.join(row_list)+'\\\\')
lines.append('\hline')

# add average values
row=['Avg.']
avg=df.mean(axis=0)
for i in avg:
	row.append(util.no_zeros_formatter(i))
row=' & '.join(row)+'\\\\'
lines.append(row)

lines.append('\hline')
lines.append('\end{tabular}')
lines.append('%%%%%%%%%%%%%%%%%%%%%%%%')
string='\n'.join(lines)
string=string.replace('nan', '---')
print('\n', string, '\n')
print('System was superior to SHR in {} of {} cases!\n'.format(outperformed,outperformed+not_outperformed))

