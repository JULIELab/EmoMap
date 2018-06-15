import prepare_data as data 
import pandas as pd
import numpy as np
import scipy.stats as st
from itertools import combinations


emotions=['Valence', 'Arousal',  'Dominance'] 

def FORMATTER(x):
	return '{:.3f}'.format(round(x,3)).lstrip('0')

def compute_isr(df1,df2):
	results={}
	for emo in emotions:
		results[emo]=None
	results['N']=None
	# get indices of both dataframes and compute intersection
	intersection = list(set(df1.index.values) & set(df2.index.values))
	results['N']=len(intersection)
	df1=df1.ix[intersection]
	df2=df2.ix[intersection]
	# drop duplicates
	df1=df1[~df1.index.duplicated(keep='first')]
	df2=df2[~df2.index.duplicated(keep='first')]
	#
	for emo in set(emotions)&set(list(df1)) & set(list(df2)):
		# compute pearsons r and write result in result's table
		results[emo]=st.pearsonr(df1[emo], df2[emo])[0]
	return results

def compute_inter_study_reliability(output_tex, output_tsv):
	isr_table = pd.DataFrame(columns=emotions+['N'])

	english={'Anew':data.get_english(),
			 'Warriner':data.load_warriner13()}

	polish={'Imbir':data.get_polish(),
			'Riegel':data.load_riegel15()}

	german={'Angst':data.get_german(),
			'Kankse':data.load_kanske10()}

	spanish={'Redondo':data.get_spanish(),
			 'Hinojosa':data.load_hinojosa16(),
			 'Stadthagen': data.load_stadthagen16(),
			 'Guasch': data.load_guasch15()}

	languages=[english, spanish, german, polish]
	for lang in languages:
		for tpl in combinations(lang,2):
			tpl=sorted(tpl)
			dataset1 = tpl[0]
			dataset2 = tpl[1]
			isr_table.loc[dataset1 +'---'+dataset2]=compute_isr(lang[dataset1], lang[dataset2])
	isr_table.N=isr_table.N.astype(int)
	isr_table.sort_values(by="Valence", axis=0, inplace=True)
	isr_table=isr_table.round(3)
	print(isr_table.to_string(float_format=FORMATTER))
	with open(output_tex, 'w') as f:
		print(isr_table.to_latex(float_format=FORMATTER), file=f)
	isr_table.to_csv(output_tsv, sep='\t')



