import pandas as pd 
from framework.util import load_tsv
from framework import prepare_data

# compute additional entries for each data set in the monolingual approach
with open('new_entries_monolingual.tsv', 'w') as f:
	print('Lexcion\tN',file=f)
	#english
	set_new=set(load_tsv('lexicons/Warriner_BE.tsv').index)
	set_old=set(prepare_data.load_anew99().index)
	n_new=len(set_new.difference(set_old))
	print('{}\t{}'.format('Warriner_BE',n_new), file=f)

	#spanish
	set_new=set(load_tsv('lexicons/Stadthagen_Dominance.tsv').index)
	set_old=set(prepare_data.load_redondo07().index)
	set_old=set(prepare_data.load_hinojosa16().index).union(set_old)
	n_new=len(set_new.difference(set_old))
	print('{}\t{}'.format('Stadthagen_Dominance',n_new), file=f)

	#german vo
	set_new=set(load_tsv('lexicons/Vo_BE.tsv').index)
	set_old=set(prepare_data.load_briesemeister11().index)
	n_new=len(set_new.difference(set_old))
	print('{}\t{}'.format('Vo_BE',n_new), file=f)

	#polish
	set_new=set(load_tsv('lexicons/Imbir_BE.tsv').index)
	set_old=set(prepare_data.load_wierzba15().index)
	n_new=len(set_new.difference(set_old))
	print('{}\t{}'.format('Imbir_BE',n_new), file=f)


# find top 10 entries per categorie in Warriner_BE
df=load_tsv('lexicons/Warriner_BE.tsv')

def reduce_margin(of, by=4):
	return '\\hspace{-'+str(by)+'pt}'+of+'\\hspace{-'+str(by)+'pt}'

def top_k_entries(df, k=10):
	### top k entries per dimension
	k=10
	df_top_k=pd.DataFrame(columns=list(df))

	for var in list(df):
		top_k=df.sort_values(var, axis=0, ascending=False)[var].head(k)
		df_top_k[var]=list(top_k.index)
	print(df_top_k)
	df_top_k.to_csv('top_10_entries_in_english_lex.tsv', sep='\t')
	###adding hspaces
	df_top_k=df_top_k.apply(reduce_margin)
	df_top_k.rename(columns=reduce_margin, inplace=True)

	string_rep= df_top_k.to_latex(index=False, escape=False)
	string_rep=string_rep.replace('\\toprule', '\\hline')
	string_rep=string_rep.replace('\\bottomrule', '\\hline')
	string_rep=string_rep.replace('\\midrule', '\\hline\\hline')
	return string_rep

print(top_k_entries(df))