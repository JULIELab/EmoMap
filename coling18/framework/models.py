import keras.models 
from keras.layers import Dense, Dropout, Activation, Input, Concatenate
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from framework import util
import os
import math


class Model():
	def __init__(self):
		raise NotImplementedError

	def fit(self):
		raise NotImplementedError

	def predict(self):
		raise NotImplementedError

class Word_Model(Model):
	def fit(self, words, labels):
		raise NotImplementedError
	def predict(self, words):
		raise NotImplementedError

def pearson(y_true, y_pred):

	fsp = y_pred - K.mean(y_pred) #being K.mean a scalar here, it will be automatically subtracted from all elements in y_pred
	fst = y_true - K.mean(y_true)

	devP = K.std(y_pred)
	devT = K.std(y_true)

	return K.mean(fsp*fst)/(devP*devT)


class SKlearn_Mapping_Model(Word_Model):
	def __init__(self, base_model, source_lexicon):
		self.model=None
		self.untrained=base_model
		self.source_lex=source_lexicon
		self.targets=None
		self.initialize()

	def initialize(self):
		self.model=clone(self.untrained)

	def __feature_extraction__(self, words):
		return np.array([self.source_lex.loc[word] for word in words])

	def fit(self, words, labels):
		# self.model.fit(features, labels)
		self.targets=labels.columns
		features=self.__feature_extraction__(words)
		self.model.fit(features, labels)

	def predict(self, words):
		features=self.__feature_extraction__(words)
		preds=self.model.predict(features)
		return pd.DataFrame(preds, columns=self.targets)


class Mapping_Model(Word_Model):
	'''
	Wrapper for Keras based MLP
	'''
	

	def __init__(	self,
					layers, #including input and output layer
					activation,
					dropout_hidden,
					batch_size,
					optimizer,
					source_lexicon,	# Emotion lexicon with entries in the 
								 	# source representation. Must also cover 
								 	# the entries in the test set.
					verbose=0,
					epochs=None,
					train_steps=None,
					kind='joint', #either "joint" or "separate"
					):
		'''
		ARGS
			batch_generation		either 'serial' or 'radomreplace'
			epochs					Will be interpreted as training steps
									if batch_generation is set to "randomreplace"
									WATCH OUT!!!
		'''
		self.targets=None
		self.source_lexicon=source_lexicon
		self.epochs=epochs
		self.train_steps=train_steps #will "round up" to make full epochs
		self.batch_size=batch_size
		self.verbose=verbose
		self.layers=layers
		self.activation=activation
		self.dropout_hidden=dropout_hidden
		self.optimizer=optimizer
		self.kind=kind
		self.kinds={'joint':self.__init_joint__, 'separate':self.__init_separate__}


		assert  (epochs is not None) or (train_steps is not None), 'Either epochs or train_streps must be set.'
		assert not( epochs is not None and train_steps is not None ), 'You cannot specify both epochs and train_steps.'

		self.initialize()

	def __init_separate__(self):
		input_layer = Input(shape=(self.layers[0],))
		top_layer=[]
		for i in range(self.layers[-1]):
			curr_layers=[input_layer]
			for j in range(len(self.layers)-2):
				curr_layers.append(Dense(self.layers[j+1])(curr_layers[-1]))
				curr_layers.append(Activation(self.activation)(curr_layers[-1]))
				curr_layers.append(Dropout(rate=self.dropout_hidden)(curr_layers[-1]))
			#last dense layer
			top_layer.append(Dense(1)(curr_layers[-1]))
		out=Concatenate()(top_layer)
		self.model=keras.models.Model(inputs=[input_layer], outputs=out)
		self.model.compile(optimizer=self.optimizer, loss='mse', metrics=[pearson])

	def __init_joint__(self):
		self.model=keras.models.Sequential()
		self.model.add(Dense(self.layers[1], input_dim=self.layers[0]))
		i=1
		while i<len(self.layers)-1:
			self.model.add(Activation(self.activation))
			self.model.add(Dropout(rate=self.dropout_hidden))
			self.model.add(Dense(self.layers[i+1]))
			i+=1
		self.model.compile(optimizer=self.optimizer, loss='mse', metrics=[pearson])

	def initialize(self):
		self.kinds[self.kind]()

	def __feature_extraction__(self, words):
		return np.array([self.source_lexicon.loc[word] for word in words])

	def fit(self, words, labels):
		self.targets=labels.columns
		features=self.__feature_extraction__(words)



		if bool(self.epochs)==True:
			if self.verbose >0:
				print('Using epoch wise training')
			self.model.fit(	features, labels, 
							epochs=self.epochs,
							batch_size=self.batch_size,
							verbose=self.verbose)
		elif bool(self.train_steps)==True:
			if self.verbose > 0:
				print('Using step-wise training.')
			bs=util.Serial_Batch_Gen(	features=pd.DataFrame(features), 
												labels=pd.DataFrame(labels),
												batch_size=self.batch_size)
			for i_step in range(self.train_steps):
				if i_step%100==0 and self.verbose>0:
					print('Now at training step: '+str(i_step))
				batch_features,batch_labels=bs.next()
				self.model.train_on_batch(batch_features,batch_labels)

		else:
			raise ValueError('Neither epochs nore train_steps are specified!')


	def predict(self,words):
		features=self.__feature_extraction__(words)
		preds=self.model.predict(features)
		return preds

	def lexicon_creation(self, words, features):
		preds=self.model.predict(features)
		return pd.DataFrame(preds, index=words, columns=self.targets)


	def test_at_steps(self, words, labels, test_split, test_steps, iterations):
		# self.targets=labels.columns
		# step 1 feature extraction

		assert bool(self.train_steps)==True, 'Training must be specified by the number of training steps'

		features=self.__feature_extraction__(words)
		labels=pd.DataFrame(labels, index=words)


		performance=pd.DataFrame(index=np.arange(1,iterations+1))
		performance.index.names=['iteration']
		for i in range(iterations):
			number_of_iteration=i+1
			# print(number_of_iteration)
			features_train, features_test,\
				labels_train,\
				labels_test=util.train_test_split(
					features, labels, test_size=test_split)
			bs=util.Serial_Batch_Gen(	features=pd.DataFrame(features_train), 
												labels=pd.DataFrame(labels_train),
												batch_size=self.batch_size)

			for i_steps in range(self.train_steps):
				total_steps=i_steps+1
				batch_features,batch_labels=bs.next()
				self.model.train_on_batch(batch_features,batch_labels)
				if total_steps in test_steps:
					preds=pd.DataFrame(self.model.predict(features_test), columns=list(labels))
					# print(preds)
					perf=np.mean(util.eval(labels_test, preds))
					performance.loc[number_of_iteration, total_steps]=perf
			# resets model
			self.initialize()
		# print(performance)
		perf_mean=pd.Series(performance.mean(axis=0))
		return perf_mean
			


class Evaluator():
	def __init__(self, models):
		'''
		ARGS
			models			A dict mapping model string identifier to and 
							instance of the model class.
		'''
		self.models=models

	def crossvalidate(self, words, labels, k_splits, outpath):
		'''
		Performs crossvalidation with each of the models given to this instance
		of the Evaluator class. The different models are tested on identical
		train/test splits which allows for using paired t-tests.
		'''
		if not os.path.isdir(outpath):
			os.makedirs(outpath)
		words=pd.Series(words)

		assert (len(words)==len(labels)), 'Words and labels have unequal'+\
			'length.'
		assert (k_splits>1), 'Crossvalidation makes no sense for k<2!'

		results={key:pd.DataFrame(columns=labels.columns)\
			for key in list(self.models)}
		k=1
		for  train_index, test_index in KFold(n_splits=k_splits, shuffle=True).\
			split(labels):
			print('k='+str(k))
			train_labels=labels.iloc[train_index]
			test_labels=labels.iloc[test_index]
			train_words=words.iloc[train_index]
			test_words=words.iloc[test_index]

			for model_name in list(self.models):
				print(model_name)
				model=self.models[model_name]
				model.fit(train_words, train_labels)
				preds=model.predict(test_words)
				preds=pd.DataFrame(preds, columns=labels.columns)
				results[model_name].loc[k]=util.eval(test_labels,preds)
				model.initialize() #resets the model.
			k+=1
		# Averaging results
		print('\n')
		for m_name in list(results):
			results[m_name]=util.average_results_df(results[m_name])
			util.save_tsv(df=results[m_name], path=outpath+'/'+m_name+'.tsv')




