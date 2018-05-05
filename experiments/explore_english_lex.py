import pandas as pd 
import numpy as np 
import scipy.stats as st 
from prepare_data import load_warriner13

df=pd.read_csv('../resources/EnglishBE.tsv', sep='\t', index_col=0)


def formatter(x):
	return '{:+.2f}'.format(round(x,2))


def reduce_margin(of, by=4):
	return '\\hspace{-'+str(by)+'pt}'+of+'\\hspace{-'+str(by)+'pt}'

def replace_rulers(string_rep):
	string_rep=string_rep.replace('\\toprule', '\\hline')
	string_rep=string_rep.replace('\\bottomrule', '\\hline')
	string_rep=string_rep.replace('\\midrule', '\\hline')
	return string_rep

def top_k_entries(k=10):
	### top k entries per dimension
	k=10
	df_top_k=pd.DataFrame(columns=list(df))

	for var in list(df):
		top_k=df.sort_values(var, axis=0, ascending=False)[var].head(k)
		df_top_k[var]=list(top_k.index)
	print(df_top_k)
	df_top_k.to_csv('../analysis/top_10_entries_in_english_lex.tsv', sep='\t')
	###adding hspaces
	df_top_k=df_top_k.apply(reduce_margin)
	df_top_k.rename(columns=reduce_margin, inplace=True)

	string_rep= df_top_k.to_latex(index=False, escape=False)
	string_rep=string_rep.replace('\\toprule', '\\hline')
	string_rep=string_rep.replace('\\bottomrule', '\\hline')
	string_rep=string_rep.replace('\\midrule', '\\hline\\hline')
	print(string_rep)


def descriptive_statistics():
	### descriptive statistics
	df_stats=pd.DataFrame(columns=list(df))
	df_stats.loc['Mean']=df.mean(axis=0)
	df_stats.loc['Median']=df.median(axis=0)
	df_stats.loc['Min']=df.min(axis=0)
	df_stats.loc['Max']=df.max(axis=0)
	df_stats.loc['StDev']=df.std(axis=0)
	df_stats=df_stats.round(2)
	df_stats.to_csv('../analysis/descriptive_statistics_english_lex.tsv',
					sep='\t')
	string_rep=df_stats.to_latex()
	string_rep=replace_rulers(string_rep)
	print(string_rep)



def correlation_matrix(df):
	warriner=load_warriner13()
	df=warriner.join(df, how='inner')
	# set lower triangular to nan
	df_corr=np.array(df.corr().astype(float))
	il1 = np.tril_indices(8)	
	df_corr[il1]=np.nan
	#
	short_dict={'Valence':'V', 'Arousal':'A', 'Dominance':'D', 
				'Joy':'J', 'Anger':'A', 'Sadness':'S', 'Fear':'F', 'Disgust':'D'}
	short_names=[short_dict[var] for var in list(df)]
	df_corr=pd.DataFrame(df_corr, index=short_names, columns=short_names)
	df_corr=df_corr.round(2)
	df_corr.to_csv('../analysis/correlation_matrix_english_lex.tsv', sep='\t')
	string_rep=df_corr.to_latex(na_rep='-', float_format=formatter)
	string_rep=string_rep.replace('-0.', '--.').replace('+0.','+.').replace('+nan',' - ')
	string_rep=string_rep.replace('\\toprule', '\\hline')
	string_rep=string_rep.replace('\\bottomrule', '\\hline')
	string_rep=string_rep.replace('\\midrule', '\\hline')
	string_rep=string_rep.replace('lrrrrrrrr', '|l|rrr|rrrrr|')
	string_list = string_rep.split('\n')
	string_list.insert(7,'\\hline')
	string_rep='\n'.join(string_list)
	print(string_rep)

top_k_entries(k=10)
descriptive_statistics()
correlation_matrix(df)