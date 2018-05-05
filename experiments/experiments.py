import prepare_data
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.svm
import scipy.stats as st
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import sys
import itertools
import re


class model_collection(object):
	'''
	This class groups functions and and models for the different emotion
	dimension/categories and provides a handy interface.
	'''

	def __init__(self, model_type):

		### Attributes
		self.VAD = ['Valence', 'Arousal',  'Dominance']
		self.BE5 = ['Joy','Anger','Sadness','Fear','Disgust']
		self.model_type =model_type
		self.models = {}
		
	def __instantiate_model__(self):
		if self.model_type=='knn':
			return sklearn.neighbors.KNeighborsRegressor(n_neighbors = 20)
		elif self.model_type=='lm':
			return sklearn.linear_model.LinearRegression()
		elif self.model_type=='rf':
			return sklearn.ensemble.RandomForestRegressor(n_estimators=500)
		elif self.model_type=='svm':
			return sklearn.svm.SVR(kernel='rbf')
		elif self.model_type=='ridge':
			return sklearn.linear_model.Ridge (alpha = .5)
		else:
			raise NotImplementedError

	def fit(self, training_df):
		for dim in self.VAD:
			self.models[dim]=self.__instantiate_model__()
			self.models[dim].fit(X=training_df[self.BE5], y=training_df[dim])
		for cat in self.BE5:
			self.models[cat]=self.__instantiate_model__()
			self.models[cat].fit(X=training_df[self.VAD], y=training_df[cat])

	def predict(self, test_df):
		preds={dimcat:None for dimcat in self.VAD+self.BE5}
		# only if every basic emotion is in test_df
		if set(self.BE5).issubset(set(test_df)):
			for dim in self.VAD:
				preds[dim]=self.models[dim].predict(X=test_df[self.BE5])
		# only if every vad dimension is in test_df
		if set(self.VAD).issubset(set(test_df)):
			for cat in self.BE5:
				preds[cat]=self.models[cat].predict(X=test_df[self.VAD])
		df = pd.DataFrame.from_dict(preds)
		df.set_index(test_df.index.values, inplace=True)
		df = df[self.VAD+self.BE5]
		return df

	def eval(self, test_df):
		preds_df = self.predict(test_df)
		pearson_coefficients = {}
		for dim in self.VAD:
			pearson_coefficients[dim] = st.pearsonr(test_df[dim],
													   preds_df[dim])[0]
		pearson_coefficients['Avg_VAD'] = np.mean([pearson_coefficients[x] for x in self.VAD])
		for cat in self.BE5:
			pearson_coefficients[cat]=st.pearsonr(test_df[cat], preds_df[cat])[0]
		pearson_coefficients['Avg_BE5'] = np.mean([pearson_coefficients[x] for x in self.BE5])
		return pearson_coefficients

	def cross_validate(self, gold_df, k=10):
		intermed_results=pd.DataFrame(columns=emotions)
		kf=KFold(n_splits=k, shuffle=True, random_state=42)
		for index,split in enumerate(kf.split(gold_df)):
			train,test=split
			# print(index)
			# print(train)
			# print(test)
			self.fit(gold_df.iloc[train]) #gets rows based on index
			intermed_results.loc[index] =self.eval(gold_df.iloc[test])
		# Average all results
		return intermed_results.mean(axis=0),intermed_results.std(axis=0, ddof=1)


def __formatter__(x):
	return '{:.3f}'.format(round(x,3)).lstrip('0')



emotions=['Valence', 'Arousal',  'Dominance', 'Avg_VAD',
		  'Joy','Anger','Sadness','Fear','Disgust', 'Avg_BE5']

data = {
		'English':prepare_data.get_english(),
		'Spanish':prepare_data.get_spanish(),
		'Polish':prepare_data.get_polish(),
		'German':prepare_data.get_german()
		}




def experiment1_monolingual(output_mean_tex,
								output_mean_tsv,
								output_std_tex,
								output_std_tsv,
								languages,
								model='knn'):
	'''Uses 10-fold CV'''
	results_mean=pd.DataFrame(columns=emotions, index=languages)
	results_std=pd.DataFrame(columns=emotions, index=languages)
	# print(results)
	for lang in languages:
		curr_model=model_collection(model_type=model)
		curr_results=curr_model.cross_validate(data[lang])
		# print(curr_results)
		# print(type(curr_results))
		mn,std=curr_model.cross_validate(data[lang])
		results_mean.loc[lang]=mn
		results_std.loc[lang]=std
	results_mean = results_mean.round(3)
	results_std=results_std.round(3)
	# output means
	print(results_mean.to_latex(float_format=__formatter__))
	with open(output_mean_tex, 'w') as f:
		print(results_mean.to_latex(float_format=__formatter__), file=f)
	with open(output_mean_tsv, 'w') as f:
		print(results_mean.to_csv(sep='\t'), file=f)
	# output standard deviations
		print(results_std.to_latex(float_format=__formatter__))
	with open(output_std_tex, 'w') as f:
		print(results_std.to_latex(float_format=__formatter__), file=f)
	with open(output_std_tsv, 'w') as f:
		print(results_std.to_csv(sep='\t'), file=f)




def experiment2_crosslingual(output_tex,
							 output_tsv,
							 languages,
							 model='knn'):
	### train collection of models for each language
	lang_codes={'English':'en', 'Spanish':'es', 'German':'de', 'Polish':'pl'}
	models ={}
	results=pd.DataFrame(columns=emotions)
	for lang in languages:
		models[lang] = model_collection(model_type=model)
		models[lang].fit(data[lang])

	### evaluate for each language pair (target not equal to source language)
	for source in languages:
		for target in languages:
			if source!=target:
				code=lang_codes[source]+'2'+lang_codes[target]
				results.loc[code]=models[source].eval(data[target])
	results = results.round(3)
	print(results.to_latex(float_format=__formatter__))
	with open(output_tex, 'w') as f:
		print(results.to_latex(float_format=__formatter__), file=f)
	with open(output_tsv, 'w') as f:
		print(results.to_csv(sep='\t'), file=f)


def experiment_3_exploration(languages, outpath):
	#enumerate all possible combinations
	combs=[]
	for i in range(2,5):
		curr_list=[]
		for j in itertools.combinations(languages, i):
			curr_list.append(list(j))
		combs.extend(curr_list)
	#setting up results table
	table=pd.DataFrame(columns=['VAD', 'BE', 'Avg'],
					   index=[' '.join(comb) for comb in combs])
	#filling table
	for comb in combs:
		curr_result=experiment3_all_vs_1_language(train_languages=comb,
												  test_languages=languages,
												  toFile=False,
												  output_tex='',
												  output_tsv='')
		averages=curr_result.mean(axis=0)
		table.loc[' '.join(comb), 'VAD']=averages['Avg_VAD']
		table.loc[' '.join(comb), 'BE']=averages['Avg_BE5']
	table['Avg']=table.mean(axis=1)
	
	# Sorts table so that best combinations of source languages appears at the
	# top
	table=table.round(3)
	table.sort_values(by='Avg', inplace=True, ascending=False)
	print(table)
	table.to_csv(outpath, sep='\t')



def experiment3_all_vs_1_language(output_tex,
								  output_tsv,
								  train_languages,
								  test_languages,
								  model='knn',
								  toFile=True):
	'''
	For each LANGUAGE in "test_languages" 
		train on all other "test_languages" 
		test on LANGUAGE
	'''
	results=pd.DataFrame(columns=emotions, index=test_languages)
	models={}
	for test_lang in test_languages:
		#build training data from all data sets other then lang
		train_langs =[x for x in train_languages if not x == test_lang]
		train_data = data[train_langs[0]]
		for next_lang in train_langs[1:]:
			train_data=train_data.append(data[next_lang])
		# create and fit models
		models[test_lang] = model_collection(model_type=model)
		models[test_lang].fit(train_data)
		results.loc[test_lang]=models[test_lang].eval(data[test_lang])
	if toFile==True:
		results = results.round(3)
		print(results.to_latex(float_format=__formatter__))
		with open(output_tex, 'w') as f:
			print(results.to_latex(float_format=__formatter__), file=f)
		with open(output_tsv, 'w') as f:
			print(results.to_csv(sep='\t'), file=f)
	else:
		return results

