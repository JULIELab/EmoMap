import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn.model_selection
import sys
import random
import os
from sklearn.metrics import mean_absolute_error as mae

# from sklearn.model_selection import KFold


### function that rescales data
def scaleInRange(x, oldmin, oldmax, newmin,newmax):
    #linear scaling (see koeper 2016). (Softmax makes no sense)
    return ((newmax-newmin)*(x-oldmin))/(oldmax-oldmin)+newmin

def scale_predictions_to_seeds(preds, seed_lexicon):
    seed_mins=seed_lexicon.min(axis=0)
    seed_maxes=seed_lexicon.max(axis=0)
    preds_mins=preds.min(axis=0)
    preds_maxes=preds.max(axis=0)

    for var in list(preds):
        preds[var]=scaleInRange(   	preds[var],
                                    oldmin=preds_mins[var],
                                    oldmax=preds_maxes[var],
                                    newmin=seed_mins[var],
                                    newmax=seed_maxes[var])
    return preds

def feature_extraction(words, embedding_model):
	'''
	Takes list of words (length m) as well as embedding model. 
	Returns pandas data frame consisting of embeddings with words as indexes
	'''
	embedding_matrix=np.zeros([len(words),embedding_model.dim])
	for i in range(len(words)):
		embedding_matrix[i,]=embedding_model.represent(words[i])
	return pd.DataFrame(data=embedding_matrix, index=words)


def eucledian_loss(true, prediction):
	'''
	True and Prediction must be tensorflow nodes. Returns another
	tf node.
	'''
	return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(true-prediction),axis=1)))


def rmse_loss(true, prediction):
	return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(true, prediction))))

def squared_error_loss(true, prediction):
	return tf.reduce_sum(tf.squared_difference(true, prediction))

def absolute_error_loss(true, prediction):
	return tf.reduce_sum(tf.abs(tf.subtract(true, prediction)))

def cosine_loss(true, prediction):
	# unit_true=tf.divide(true, tf.sqrt(tf.reduce_sum(tf.square(true), 1, keep_dims=True)))
	# unit_prediction=tf.divide(true, tf.sqrt(tf.reduce_sum(tf.square(true), 1, keep_dims=True)))
	# return tf.losses.cosine_distance(tf.divide(true, tf.norm(true)),
	# 						  tf.divide(prediction, tf.norm(prediction)), dim=1)
	enumerator=tf.reduce_sum(tf.multiply(true,prediction), axis=1, keep_dims=True)
	denominator=tf.multiply(tf.norm(true,axis=1, keep_dims=True), tf.norm(prediction, axis=1, keep_dims=True))	
	cos=tf.divide(enumerator,denominator)
	return tf.reduce_sum(1.-cos, axis=0)

def leaky_relu(x, alpha=.01, name=None):
	assert alpha<1 and alpha>0, 'Alpha is supposed to be between 0 and 1'+\
		' (typically .01).'
	return tf.maximum(x, alpha * x, name=name)




def combine_features_labels (features, labels):
		assert features.shape[0]==labels.shape[0], 'Features and labels differ in length!'
		features.set_index([list(range(features.shape[0]))], inplace=True)
		labels.set_index([list(range(labels.shape[0]))], inplace=True)
		data=pd.concat([features,labels], axis=1)
		# selection of features and training data correct?
		feature_cols=[0,features.shape[1]]
		label_cols=[features.shape[1],data.shape[1]]
		return data, feature_cols, label_cols


def rmse(a,b):
	return np.sqrt(np.mean(np.power(a-b,2)))

# def mae(a,b):
# 	return np.mean(np.absolute(a-b))

def eval(true, prediction, metric='r'):
	# TODO: write Tests!
	'''
	Expects pandas data frames.
	'''
	metrics={	'r':lambda x,y:st.pearsonr(x,y)[0],
				'rmse':lambda x,y: np.sqrt(np.mean(np.power(x-y,2))),
				'mae':mae #lambda x,y: np.mean(np.absolute(x-y))
			}
	metric=metrics[metric]
	row=[]
	for var in list(prediction):
		value=metric(prediction[var], true[var])
		# print(value)
		row+=[value]
	return row




class Batch_Gen():
	def __init__(self, features, labels, batch_size):
		self.data,\
		self.feature_cols,\
		self.label_cols=combine_features_labels(features, labels)
		self.batch_size=batch_size
		self.pointer=0
		self.len=len(self.data)
	def next(self):
		raise NotImplementedError('This is supposed to be an abstract method.')


class Serial_Batch_Gen(Batch_Gen):
	def next(self):
		if self.pointer+self.batch_size<self.len:
			features=self.data.iloc[self.pointer:self.pointer+self.batch_size,\
									self.feature_cols[0]:self.feature_cols[1]]
			labels=self.data.iloc[self.pointer:self.pointer+self.batch_size,\
									self.label_cols[0]:self.label_cols[1]]
			# features=self.sub_data.iloc[:,self.feature_cols]
			# labels=self.sub_data.iloc[:,self.label_cols]
			self.pointer=self.pointer+self.batch_size	
		else:
			# restart from beginning
			rest=self.pointer+self.batch_size-self.len
			features=self.data.iloc[self.pointer:,\
									self.feature_cols[0]:self.feature_cols[1]]
			labels=self.data.iloc[self.pointer:,\
									self.label_cols[0]:self.label_cols[1]]
			rest_df=self.data.iloc[:rest,:]
			rest_features=rest_df.iloc[:rest,\
										self.feature_cols[0]:self.feature_cols[1]]
			rest_labels=rest_df.iloc[:rest,\
										self.label_cols[0]:self.label_cols[1]]
			labels=pd.concat([labels, rest_labels])
			features=pd.concat([features, rest_features])
			self.pointer=rest
			#	
		return features, labels

class Random_Replace_Batch_Gen(Batch_Gen):
	def next(self):
		sample=[]
		population=list(range(self.len))
		for __ in range(self.batch_size):
			sample.append(random.choice(population))
		features=self.data.iloc[sample, self.feature_cols[0] : self.feature_cols[1]]
		labels=self.data.iloc[sample, self.label_cols[0] : self.label_cols[1]]
		return features, labels

class Random_Batch_Gen(Batch_Gen):
	def next(self):
		bools=[True]*self.batch_size+[False]*(self.len-self.batch_size)
		np.random.shuffle(bools)
		features=self.data.loc[bools].iloc[:, self.feature_cols[0]:self.feature_cols[1]]
		labels=self.data.loc[bools].iloc[:, self.label_cols[0]:self.label_cols[1]]
		return features, labels



def train_test_split(features, labels, test_size):
    features_train, features_test, labels_train, labels_test=sklearn.model_selection.train_test_split(features, labels, test_size=test_size)
    return features_train, features_test, labels_train, labels_test

def k_folds_split(features, labels, k):
	folds=[]
	data, feature_cols, label_cols=combine_features_labels(features, labels)
	kf=sklearn.model_selection.KFold(n_splits=k, shuffle=True)
	for __, split in enumerate(kf.split(data)):
		train_data=data.iloc[split[0]]
		features_train=train_data.ix[:,feature_cols[0]:feature_cols[1]]
		labels_train=train_data.ix[:,label_cols[0]:label_cols[1]]
		#
		test_data=data.iloc[split[1]]
		features_test=test_data.ix[:,feature_cols[0]:feature_cols[1]]
		labels_test=test_data.ix[:,label_cols[0]:label_cols[1]]
		folds+=[(features_train, labels_train, features_test, labels_test)]
	return iter(folds)

def average_results_df(results_df):
	avg=results_df.mean(axis=0)
	sd=results_df.std(axis=0)
	results_df.loc['SD']=sd
	results_df.loc['Average']=avg
	results_df['Average']=results_df.mean(axis=1)
	#results_df.loc['SD']=
	return results_df

def get_average_result_from_df(file):
	df=pd.read_csv(file, sep='\t')
	# print(df)
	# print(df.columns)
	df=df.rename(columns = {'Unnamed: 0':'k'})
	# df.columns=['k']+df.columns[1:]
	# df.columns=['k', 'Valence', 'Arousal', 'Dominance', 'Average']
	df.set_index('k', inplace=True)
	return df.ix['Average','Average']

# def write_readme(path, text):
# 	import datetime
# 	with open(path+'README.md','w') as README:
# 		README.write(text)
# 		README.write('\n\nRan on '+str(datetime.date.today())+'\n')

def load_tsv(path):
	return pd.read_csv(path, sep='\t', index_col=0)

def save_tsv(df, path, dec=4):
	folder='/'.join(path.split('/')[:-1])
	# if not folder=='':	
	# 	if not os.path.isdir(folder):
	if bool(folder)==True and  (not os.path.isdir(folder)):
		os.makedirs(folder)
	df=df.round(dec)
	# df.to_csv(path, sep='\t', float_format='%.3f') # '{:.3f}'
	with open(path, 'w') as f:
		print(df.to_csv(sep='\t'), file=f)

def no_zeros_formatter(x):
	return '{:.3f}'.format(round(x,3)).lstrip('0').replace('-0.', '-.')

def err_print(*args, **kwargs):
	###prints to stderr
    print(*args, file=sys.stderr, **kwargs)

def split_df(df, portion):
	num=int(len(df)*portion)
	select=np.array([True]*num+[False]*(len(df)-num))
	np.random.shuffle(select)
	df1=df[select]
	df2=df[~select]
	return df1, df2

def drop_duplicates(df):
	df=df[~df.index.duplicated(keep='first')]
	return df

def average_subdirs(parent):
	'''
	Assuming a parent directory with n subdirs each of them having identically
	named files in them. Averages the data frame saved in the identically
	named files in each sudir.

	The new data frames are written into a sudir "Average" (this dir is ignored
	in the above process.)
	'''
	content=os.listdir(parent)
	if 'Average' in content: 
		content.remove('Average')
	print(content)
	dirs=[parent+'/'+x for x in content if os.path.isdir(parent+'/'+x)]
	# print(dirs)
	files=[dirs[0]+'/'+x for x in os.listdir(dirs[0]) if os.path.isfile(dirs[0]+'/'+x) ]

	def name(path):
		return path.split('/')[-1]

	def names(paths):
		return [name(x) for x in paths]

	# print(files)

	### build up emptpy list for each shared filename
	dfs={name:[] for name in names(files)}

	### check if all subdirs have identical files
	### populate empty lists
	for d in dirs:
		curr_files=[d+'/'+x for x in os.listdir(d) if os.path.isfile(d+'/'+x)]
		if not names(files)==names(curr_files):
			print(curr_files)
			print(files)
			raise ValueError('Sudirectories do not have identical content!')
		for f in curr_files:
			df=load_tsv(f)
			dfs[name(f)].append(df)

	###compute averages
	avg_dfs={}
	for name in list(dfs):
		curr_dfs=dfs[name]
		#convert into dict
		curr_dfs={i:curr_dfs[i] for i in range(len(curr_dfs))}
		avg_dfs[name]=pd.Panel(curr_dfs).mean(axis=0)
	# print(avg_dfs)

	### save averages in new "average" folder
	outpath=parent+'/Average'
	if not os.path.isdir(outpath):
		os.makedirs(outpath)
	for name, df in avg_dfs.items():
		save_tsv(df=df, path=outpath+'/'+name)

def get_columns_if_existing(df, cols):
	cols=[c for c in cols if c in df.columns]
	return df[cols]

def compute_isr(lex1, lex2):
	words=list(set(lex1.index).intersection(set(lex2.index)))
	variables=list(set(list(lex1)).intersection(set(list(lex2))))
	lex1=lex1.loc[words,variables]
	lex2=lex2.loc[words, variables]
	results=eval(lex1,lex2)
	results=pd.Series(results, variables)
	results['N']=len(words)
	return results
