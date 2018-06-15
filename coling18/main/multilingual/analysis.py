import framework.util as util
from main.data import IN_PAPER_NAMES, VA, BE5, SHORT_COLUMNS
import datetime
import pandas as pd


df=util.load_tsv('results.tsv')


df=df[VA+BE5]
df.rename(index=IN_PAPER_NAMES,  inplace=True)
df.rename(index=str, columns=SHORT_COLUMNS, inplace=True)

#reoder index
df=df.reindex([value for key,value in IN_PAPER_NAMES.items()])



df_shr=util.load_tsv('../../analysis/shr/shr_normalized.tsv').drop('Dom', axis=1)

df=df.round(3)

df_lesser=df<df_shr
df_greater=df>df_shr




lines=[]
lines.append('%%%%%% Automatic Python output from {} &%%%%%%%%%%'.format(datetime.datetime.now()))
lines.append('\\begin{tabular}{|l|rr|rrrrr|}')
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
print('\n', string, '\n')
