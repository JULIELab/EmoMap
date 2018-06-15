import numpy as np
import pandas as pd 
from main.data import SETTINGS, IN_PAPER_NAMES
from framework.util import get_average_result_from_df, save_tsv, no_zeros_formatter, load_tsv
import datetime
import scipy.stats as st

directions=['be2vad', 'vad2be']

models=['baseline', 'reference_LM', 'Reference_KNN', 'my_model']

df=pd.DataFrame(index=[setting.name for setting in SETTINGS],
				columns=['be2vad_'+model for model in models]+\
							['vad2be_'+model for model in models])


for d in directions:
	for setting in SETTINGS:
		for model in models:
			perf=get_average_result_from_df('results/{}/{}/{}.tsv'.format(
				d, setting.name, model))
			df.loc[setting.name, d+'_'+model]=perf




df.loc['Average']=df.mean(axis=0)


df.rename(index=IN_PAPER_NAMES, inplace=True)

save_tsv(df, 'overview.tsv', dec=3)

string = df.to_latex(float_format=no_zeros_formatter)

lines=string.split('\n')
lines[0]='\\begin{tabular}{|l|rrrr|rrrr|}'
lines=['%%%%%% Automatic Python output from {} &%%%%%%%%%%'.format(datetime.datetime.now())]+lines
lines[-1]='%%%%%%%%%%%%%%%%%%%%%%%%'

lines.insert(3, '{} & \multicolumn{4}{c|}{BE2VAD} & \multicolumn{4}{c|}{VAD2BE} \\\\')
lines[4]=lines[4].replace('be2vad\_','').replace('vad2be\_', '').\
	replace('Reference\_','').replace('reference\_','').replace('my\_model','FFNN').\
	replace('baseline','Baseline')
lines[2]='\\hline'
lines[5]='\\hline\\hline'

lines[-3]='\\hline'
lines.insert(-4, '\\hline')


string='\n'.join(lines)
print(string)

with open('overview.tex', 'w') as f:
	print(string, file=f)


####################################################

### Significance tests
settings=[s.name for s in SETTINGS]


star_df=pd.DataFrame(columns=directions)

for d in directions:
	for s in settings:
		### load all individual data frames
		dfs={}
		for m in models:
			dfs[m]=load_tsv('results/{}/{}/{}.tsv'.format(d,s,m))
		# write average results into single data frame to determine the two best systems
		average_results=pd.DataFrame(columns=['r'])
		for key,value in dfs.items():
			average_results.loc[key,'r']=value.loc['Average', 'Average']
		# sort by performance and get name of the best two systems
		average_results=average_results.sort_values(by='r', axis=0, ascending=False)
		best_2=list(average_results.index)[:2]
		# compute paired t-test on individual results of cross-validation
		pvalue=st.ttest_rel(a=dfs[best_2[0]].drop(['SD','Average'], axis=0)['Average'],
							b=dfs[best_2[1]].drop(['SD','Average'], axis=0)['Average'])[1]
		# compute the number of stars
		siglevel=''
		if pvalue >=.05:
			siglevel='–'
		elif pvalue <.05 and pvalue >=.01:
			siglevel='*'
		elif pvalue <.01 and pvalue >=.001:
			siglevel='**'
		else:
			siglevel='***'
		# save siglevel in respective data frame
		star_df.loc[s,d]=siglevel
star_df.rename(index=IN_PAPER_NAMES, inplace=True)
save_tsv(star_df, 't-tests.tsv')
print(star_df)
			





