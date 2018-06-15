from  sklearn.neural_network import MLPRegressor as mlp
from sklearn.ensemble import AdaBoostRegressor as adaboost
import pandas as pd
import scipy.stats as st
import numpy as np
from framework import util
from framework.models import Word_Model


'''
Reimplementation Aicyber's contribution to the IALP 2016 shared task on 
dimensional sentiment analysis of Chinese words. See: 

	@inproceedings{du_aicybers_2016,
		title = {Aicyber's system for {IALP} 2016 shared task: {Character}-enhanced word vectors and {Boosted} {Neural} {Networks}},
		booktitle = {Asian {Language} {Processing} ({IALP}), 2016 {International} {Conference} on},
		author = {Du, Steven and Zhang, Xi},
		year = {2016},
		pages = {161--163}
	}

The original paper also applied some "special sauce" to the embedding vectors.
However, I consider this to be out of scope for our comparison (thus, we are
restricting ourselves to the proposed regressor).

They use a boosted MLP approach using the scikit learn implementation with the 
following parameters:

			paper 								code
MLP 		1 hidden layer (100 units)
base esti.	relu activation
			adam
			constant learning rate of 1e-3
												early stopping=True
												max_iter=2000

Boosting	AdaBoost
			30 estimators
			learning rate of 1e-2


'''


class MLP_Ensemble(Word_Model):

	def __init__(self, embeddings):
		# self.model=adaboost(		base_estimator=self.base_estimator)
		self.models=None # dictionary mapping from label column name to model
		self.embeddings=embeddings
		self.targets=None

	def __get_base_estimator__(self):
		return mlp(	hidden_layer_sizes=(100), 
									activation='relu',
									solver='adam',
									learning_rate='constant',
									learning_rate_init=1e-3,
									early_stopping=True,
									max_iter=2000)

	def __get_ensemble__(self):
		return adaboost(base_estimator=self.__get_base_estimator__(),
						n_estimators=30, learning_rate=1e-2)

	def __feature_extraction__(self, words):
		return np.array([self.embeddings.represent(w) for w in words])

	def initialize(self):
		self.models={target:self.__get_ensemble__() for target in self.targets}

	def fit(self, words, labels):
		# self.model.fit(features, labels)
		self.targets=labels.columns
		self.initialize()
		features=self.__feature_extraction__(words)
		for target in self.targets:
			self.models[target].fit(features, labels[target])

	def predict(self, words):
		features=self.__feature_extraction__(words)
		df=pd.DataFrame(columns=self.targets, index=words)
		for target in self.targets:
			df.loc[:,target]=self.models[target].predict(features)
		# return self.model.predict(features)
		return df

	# def eval(self, train_features, train_labels, test_features, test_labels):
	# 	self.fit(train_features, train_labels)
	# 	preds=pd.DataFrame(data=self.predict(test_features), 
	# 			index=test_features.index, columns=list(test_labels))
	# 	performance=pd.Series(index=list(test_labels)+['Average'])
	# 	for var in list(test_labels):
	# 		performance.loc[var]=st.pearsonr(preds.loc[:,var], test_labels.loc[:,var])[0]
	# 	performance.loc['Average']=np.mean(performance[:-1])
	# 	return performance

	# def crossvalidate(self, features, labels, k_folds):
	# 	k=0
	# 	results_df=pd.DataFrame(columns=labels.columns)
	# 	for fold in util.k_folds_split(features, labels, k=k_folds):
	# 		k+=1
	# 		print(k)
	# 		results_df.loc[k]=self.eval(*fold)
	# 		print(results_df)
	# 	results_df=util.average_results_df(results_df)
	# 	return results_df
