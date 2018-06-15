import numpy as np 
import pandas as pd 
import os

from main.data import MY_MODEL, VAD, VA, BE5, SETTINGS, LANGUAGES
from framework.util import save_tsv, load_tsv, drop_duplicates
from framework import prepare_data
from framework.models import Mapping_Model




### get complete data sets language wise
lang_data={}
for l in LANGUAGES:
	lang_data[l]=[]
	for s in SETTINGS:
		if s.language==l:
			data=s.load_data()
			data.index=[s.name+'_'+word for word in data.index]
			lang_data[l].append(data)
	lang_data[l]=pd.concat(lang_data[l])


def get_model(n_inputs, n_outputs, source_lexicon):
	return Mapping_Model(
			layers=[n_inputs]+MY_MODEL['hidden_layers']+[n_outputs],
			activation=MY_MODEL['activation'],
			dropout_hidden=MY_MODEL['dropout_hidden'],
			train_steps=MY_MODEL['train_steps'],
			batch_size=MY_MODEL['batch_size'],
			optimizer=MY_MODEL['optimizer'],
			source_lexicon=source_lexicon, 
			verbose=0)
	



def monolingual():

	def create_lexicon(source_rep, target_rep, train_data, target_data):
		model=get_model(n_inputs=len(source_rep), n_outputs=len(target_rep),
						source_lexicon=train_data[source_rep])
		model.fit(words=train_data.index, labels=train_data[target_rep])
		lex=model.lexicon_creation(words=target_data.index, features=target_data[source_rep])
		return lex



	#### for the monolingual set-up, we selected the data sets with the
	#### highest accuracy in the monolingual evaluation (estimated to be the cleanest data)

	settings=[	{	'name':'Warriner_BE', 
					'source_rep':VAD, 
					'target_rep':BE5, 
					'train_data':prepare_data.get_english_anew(), 
					'target_data':prepare_data.load_warriner13},

				{	'name':'Stadthagen_Dominance',
					'source_rep':BE5,
					'target_rep':VAD,
					'train_data':prepare_data.get_spanish_hinojosa(),
					'target_data':prepare_data.load_stadthagen17},

				{	'name':'Vo_BE',
					'source_rep':VA,
					'target_rep':BE5,
					'train_data':prepare_data.get_german_bawl(),
					'target_data':prepare_data.load_vo09},

				{	'name':'Imbir_BE',
					'source_rep':VA,
					'target_rep':BE5,
					'train_data':prepare_data.get_polish_nawl(),
					'target_data':prepare_data.load_imbir16}
				]

	for s in settings:
		lex=create_lexicon(	source_rep=s['source_rep'],
						target_rep=s['target_rep'],
						train_data=s['train_data'],
						target_data=s['target_data']()
						)
		save_tsv(lex, 'lexicons/{}.tsv'.format(s['name']))


#####
def crosslingual():
	all_data=pd.concat([df for key,df in lang_data.items()])

	### VA-->BE5 model on all_data
	model=get_model(n_inputs=len(VA), n_outputs=len(BE5), source_lexicon=all_data[VA])
	model.fit(words=all_data.index, labels=all_data[BE5])

	###combining chinese data
	def get_zh():
		zh1=prepare_data.load_yu16()
		zh2=prepare_data.load_yao16()
		zh=pd.concat([zh1, zh2])
		del zh1
		del zh2
		return drop_duplicates(zh)

	settings=	[ 	{'name':'it_Montefinese_BE', 'load':prepare_data.load_montefinese14},
					{'name':'pt_Soares_BE', 'load':prepare_data.load_soares12},
					{'name':'nl_Moors_BE', 'load':prepare_data.load_moors13},
					{'name':'id_Sianipar_BE', 'load':prepare_data.load_sianipar16},
					{'name':'zh_Yu_Yao_BE', 'load':get_zh},
					{'name':'fr_Monnier_BE', 'load':prepare_data.load_monnier14},
					{'name':'gr_Palogiannidi_BE', 'load':prepare_data.load_palogiannidi16},
					{'name':'fn_Eilola_BE', 'load':prepare_data.load_eilola10},
					{'name':'sv_Davidson_BE', 'load':prepare_data.load_davidson14}
				]
	num_of_new_entries=pd.DataFrame(columns=['N'])
	for s in settings:
		print(s['name'])
		source_lex=s['load']()
		lex=model.lexicon_creation(words=source_lex.index, features=source_lex[VA])
		save_tsv(lex, 'lexicons/{}.tsv'.format(s['name']))
		num_of_new_entries.loc[s['name']]=len(lex)
	save_tsv(num_of_new_entries, 'new_entries_crosslingual.tsv')



if __name__=='__main__':
	if not os.path.isdir('lexicons'):
		os.makedirs('lexicons')

	monolingual()
	crosslingual()





