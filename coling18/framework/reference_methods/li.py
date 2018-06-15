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
from naacl.framework import util
import pandas as pd


'''
Main approach advocated in 

	@article{li_inferring_2017,
		title = {Inferring {Affective} {Meanings} of {Words} from {Word} {Embedding}},
		volume = {PP},
		doi = {10.1109/TAFFC.2017.2723012},
		number = {99},
		journal = {IEEE Transactions on Affective Computing},
		author = {Li, Minglei and Lu, Qin and Long, Yunfei and Gui, Lin},
		year = {2017},
		pages = {1--1},
		}

which is mainly ridge regression on top of the embedding vectors with the 
scikit-learn default parameters.
'''

class Multi_Target_Regressor():

	def __init__(self, init_fun=sklearn.linear_model.Ridge):
		'''
		init_fun			A scikit-learn instantiation of a model such as
							linear regression or ridge regression.
		'''
		self.model=init_fun()
		self.var_names=None

	def fit(self, features, labels):
		self.model.fit(features, labels)
		self.var_names=labels.columns

	def predict(self, features):
		preds=pd.DataFrame(self.model.predict(features), index=features.index,
			columns=self.var_names)
		return preds

	def eval(self, train_features, train_labels, test_features, test_labels):
		'''
		Assumes pandas data frames as input
		'''
		self.model.fit(train_features, train_labels)
		preds=pd.DataFrame(data=self.model.predict(test_features), 
				index=test_features.index, columns=list(test_labels))
		performance=pd.Series(index=list(test_labels)+['Average'])
		for var in list(test_labels):
			# print(var)
			# print(performance)
			# print(performance.loc[var])
			performance.loc[var]=st.pearsonr(preds.loc[:,var], test_labels.loc[:,var])[0]
		performance.loc['Average']=np.mean(performance[:-1])
		return performance

	def crossvalidate(self, features, labels, k_folds):	
		k=0
		results_df=pd.DataFrame(columns=labels.columns)
		for fold in util.k_folds_split(features, labels, k=k_folds):
			k+=1
			print(k)
			#features_train, labels_train, features_test, labels_test=fold
			# define model
			# define_model(layers=layers,
			# 				   nonlinearity=nonlinearity,
			# 				   weights_sd=weights_sd,
			# 				   biases=biases)
			# define_loss(loss_function=tf.losses.mean_squared_error,
			# 		  l2_beta=0)
			# define_optimization(learning_rate=learning_rate)
			# with tf.Session() as sess:
			# 	init_session(sess)
			# 	train(session=sess, 
			# 				features=features_train, 
			# 				labels=labels_train, 
			# 				training_steps=training_steps, 
			# 				batch_size=batch_size,
			# 				dropout_hidden=dropout_hidden,
			# 				dropout_embedding=dropout_embedding, 
			# 				report_at=report_at)
			results_df.loc[k]=self.eval(*fold)
		results_df=util.average_results_df(results_df)
		return results_df


# class Model_Collection(object):
# 	'''
# 	This class groups functions and and models for the different emotion
# 	dimension/categories and provides a handy interface.
# 	'''

# 	def __init__(self, model_type):

# 		### Attributes
# 		self.VAD = ['Valence', 'Arousal',  'Dominance']
# 		self.BE5 = ['Joy','Anger','Sadness','Fear','Disgust']
# 		self.model_type =model_type
# 		self.models = {}
		
# 	def __instantiate_model__(self):
# 		if self.model_type=='knn':
# 			return sklearn.neighbors.KNeighborsRegressor(n_neighbors = 20)
# 		elif self.model_type=='lm':
# 			return sklearn.linear_model.LinearRegression()
# 		elif self.model_type=='rf':
# 			return sklearn.ensemble.RandomForestRegressor(n_estimators=500)
# 		elif self.model_type=='svm':
# 			return sklearn.svm.SVR(kernel='rbf')
# 		elif self.model_type=='ridge':
# 			return sklearn.linear_model.Ridge (alpha = .5)
# 		else:
# 			raise NotImplementedError

# 	def fit(self, features, labels):
# 		'''
# 		Assumes "features" and "labels" to be pandas data frames-
# 		'''
# 		for var in list(labels):
# 			self.models[var]=self.__instantiate_model__()
# 			self.models[var].fit(X=features, y=labels)
# 		# for dim in self.VAD:
# 		# 	self.models[dim]=self.__instantiate_model__()
# 		# 	self.models[dim].fit(X=training_df[self.BE5], y=training_df[dim])
# 		# for cat in self.BE5:
# 		# 	self.models[cat]=self.__instantiate_model__()
# 		# 	self.models[cat].fit(X=training_df[self.VAD], y=training_df[cat])

# 	def predict(self, features):
# 		# preds={}
# 		# for dim in self.VAD:
# 		# 	preds[dim]=self.models[dim].predict(X=test_df[self.BE5])
# 		# for cat in self.BE5:
# 		# 	preds[cat]=self.models[cat].predict(X=test_df[self.VAD])
# 		# df = pd.DataFrame.from_dict(preds)
# 		# df.set_index(test_df.index.values, inplace=True)
# 		# df = df[self.VAD+self.BE5]
# 		# return df

# 		preds=pd.DataFrame(index=features.index, columns=list(self.models))
# 		for var in list(self.models):
# 			preds[var]=self.models[var].predict(X=features)
# 		return preds


# 		# preds={dimcat:None for dimcat in self.VAD+self.BE5}

# 		# # only if every basic emotion is in test_df
# 		# if set(self.BE5).issubset(set(test_df)):
# 		# 	for dim in self.VAD:
# 		# 		preds[dim]=self.models[dim].predict(X=test_df[self.BE5])
# 		# # only if every vad dimension is in test_df
# 		# if set(self.VAD).issubset(set(test_df)):
# 		# 	for cat in self.BE5:
# 		# 		preds[cat]=self.models[cat].predict(X=test_df[self.VAD])
# 		# df = pd.DataFrame.from_dict(preds)
# 		# df.set_index(test_df.index.values, inplace=True)
# 		# df = df[self.VAD+self.BE5]
# 		# return df

# 	def eval(self, test_df):
# 		preds_df = self.predict(test_df)
# 		pearson_coefficients = {}
# 		for dim in self.VAD:
# 			pearson_coefficients[dim] = st.pearsonr(test_df[dim],
# 													   preds_df[dim])[0]
# 		pearson_coefficients['Avg_VAD'] = np.mean([pearson_coefficients[x] for x in self.VAD])
# 		for cat in self.BE5:
# 			pearson_coefficients[cat]=st.pearsonr(test_df[cat], preds_df[cat])[0]
# 		pearson_coefficients['Avg_BE5'] = np.mean([pearson_coefficients[x] for x in self.BE5])
# 		return pearson_coefficients

# 	def cross_validate(self, gold_df, k=10):
# 		#intermed_results=pd.DataFrame(columns=VAD+BE5)
# 		intermed_results=pd.DataFrame(columns=emotions)
# 		kf=KFold(n_splits=k, shuffle=True, random_state=42)
# 		for index,split in enumerate(kf.split(gold_df)):
# 			train,test=split
# 			# print(index)
# 			# print(train)
# 			# print(test)
# 			self.fit(gold_df.iloc[train]) #gets rows based on index
# 			intermed_results.loc[index] =self.eval(gold_df.iloc[test])
# 		# Average all results
# 		return intermed_results.mean(axis=0),intermed_results.std(axis=0, ddof=1)
