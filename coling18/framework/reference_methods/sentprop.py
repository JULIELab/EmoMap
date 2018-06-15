import numpy as np
import pandas as pd 
import scipy.stats as st
from naacl.framework import util
from sklearn.model_selection import KFold
from scipy import sparse
# from naacl.framework.representations.embedding import Embedding

class Bootstrapper():

	def __init__(self, beta, neutral_value):
		self.beta=beta # .1 works, hamilton uses .9. In our previous exp, .5 was best
		self.neutral_value=neutral_value

	def run(self, embeddings, seeds, targets,):
		'''
		Args:
		embeddings			embedding model (object)
		seeds 				pandas data frame. words in index, emotional 
							variables as column names.
		targets 			list of target words
		neutral_value		float. what is the default value for the emotional
							variables. (Assumes all variables have equal 
							default value).
		'''
		# print(seeds)
		# print(targets)

		#operate on only the words which are present in either seeds or targets
		# only if they are also in the embedding space!
		graph_words=list(set(seeds.index).union(set(targets)))
		graph_words=list(set(graph_words).intersection(set(embeddings.iw)))
		
		print('no graph_words: ', len(graph_words))

		transition_matrix=self.get_transition_matrix(
										embeddings.subsample(graph_words))
		print('transition_matrix:\n ', transition_matrix)

		seed_emotion_matrix=self.get_seed_emotion_matrix(	words=graph_words,
															seed_labels=seeds,
															neutral_value=self.neutral_value)
		
		print('seed_emotion_matrxi: \n',seed_emotion_matrix)
		current_emotion_matrix=np.ones([len(graph_words), len(seeds.columns)])*\
															self.neutral_value
		print('current_emotion_matrix.\n', current_emotion_matrix)
		diff=np.sum(current_emotion_matrix)
		iterations=0
		while diff>1e-6 and iterations<200:
			# print(iterations, diff)
			# last_emotion_matrix=np.array(current_emotion_matrix)
			new_emotion_matrix= (self.beta*(transition_matrix.dot(current_emotion_matrix))) + ((1-self.beta)*seed_emotion_matrix)
			# print('new_emo_matrix:\n ', new_emotion_matrix)
			diff=np.sum(abs(new_emotion_matrix-current_emotion_matrix))
			iterations+=1
			current_emotion_matrix=new_emotion_matrix
		print("Final difference / iterations:\t" + str(diff) + "\t" + str(iterations))
		final_df=pd.DataFrame(current_emotion_matrix, index=graph_words, 
							columns=list(seeds))
		# final_df=final_df.loc[targets,:]
		
		#add neutral valued entries for target words which are not in graph words
		for t in targets:
			if not t in final_df.index:
				final_df.loc[t]=[self.neutral_value]*final_df.shape[1]
		print('final_df:\n', final_df)
		return final_df


	def eval(self, train_labels, test_labels, embeddings):
		preds=self.run(	embeddings=embeddings, 
						seeds=train_labels,
						targets=list(test_labels.index))
		preds=preds.loc[test_labels.index, :]
		performance=pd.Series(index=list(test_labels)+['Average'])
		for var in list(test_labels):
			performance.loc[var]=st.pearsonr(preds.loc[:,var], test_labels.loc[:,var])[0]
		performance.loc['Average']=np.mean(performance[:-1])
		return performance

	def crossvalidate(self, labels, embeddings, k_folds):
		k=0
		results_df=pd.DataFrame(columns=labels.columns)
		kf=KFold(n_splits=k_folds, shuffle=True)
		for train_index, test_index in kf.split(labels):
			k+=1
			print(k)
			results_df.loc[k]=self.eval(train_labels=labels.iloc[train_index,:],
										test_labels=labels.iloc[test_index,:],
										embeddings=embeddings)
			print(results_df)
		# for fold in util.k_folds_split(features, labels, k=k_folds):
		# 	k+=1
		# 	print(k)
		# 	results_df.loc[k]=self.eval(*fold)
		# 	print(results_df)
		results_df=util.average_results_df(results_df)
		return results_df



	def get_seed_emotion_matrix(self, words, seed_labels, neutral_value):
		# # # one column per VAD dimension
		# # # invert switches order of a VAD dimension (for negative seeds)
		# seed_matrix = np.ones((len(words),3))*neutralValue
		# for i in range(len(words)):
		#	  #print(words[i])
		#	  if words[i] in seed_lexicon.index:
		#			seed_matrix[i,:]=seed_lexicon.loc[words[i]]
		# return seed_matrix
		
		# df=pd.DataFrame(index=words, columns=seed_labels.columns)
		# print(df)
		# print(seed_labels)
		# for w in words:
		# 	if w in list(seed_labels.index):
		# 		print(w, type(w))
		# 		df.loc[w]=seed_labels.loc[w]
		# 	else: 
		# 		df.loc[w]=[neutral_value]*len(seed_labels.columns)

		# for i in range(len(words)):
		# 	if words[i] in seed_labels.index:
		# 		print(words[i],i)
		# 		value=seed_labels.loc[words[i]]
		# 		df.iloc[i]=value
		# 	else:
		# 		df.iloc[i]=[neutral_value]*seed_labels.shape[1]

		# return df.as_matrix()
		# print(seed_labels.loc['soothe'])
		m=np.ones(shape=[len(words), len(seed_labels.columns)])*neutral_value
		for i in range(len(words)):
			if words[i] in seed_labels.index:
				# print(words[i])
				m[i,:]=seed_labels.loc[words[i]]
		return m



	############################################################
	'''
	Taken from socialsent.
	This should be allright, since it does not depend on the one the shape of the
	input seed lexicon.

	TODO: reimplement those when publishing toolkit!

	# '''
	def get_transition_matrix(self, embeddings, word_net=False, first_order=False, sym=False, trans=False, **kwargs):
	    """
	    Build a probabilistic transition matrix from word embeddings.
	    """
	    if word_net:
	        L =  wordnet_similarity_matrix(embeddings)
	    elif not first_order:
	        L = self.similarity_matrix(embeddings, **kwargs)
	    else:
	        L = embeddings.m.todense().A
	    if sym:
	        Dinv = np.diag([1. / np.sqrt(L[i].sum()) if L[i].sum() > 0 else 0 for i in range(L.shape[0])])
	        return Dinv.dot(L).dot(Dinv)
	    else:
	        print('foo')
	        Dinv = np.diag([1. / L[i].sum() for i in range(L.shape[0])])
	        L = L.dot(Dinv)
	    if trans:
	        return L.T
	    return L

	def similarity_matrix(self, embeddings, arccos=False, similarity_power=1, nn=25, **kwargs):
		"""
		Constructs a similarity matrix from embeddings.
		nn argument controls the degree.
		"""
		def make_knn(vec, nn=nn):
			  vec[vec < vec[np.argsort(vec)[-nn]]] = 0
			  return vec
		#print(embeddings.m[1:10,])
		# es gibt keine embeddings
		L = embeddings.m.dot(embeddings.m.T)
		if sparse.issparse(L):
			L = L.todense()
		if arccos:
			L = np.arccos(np.clip(-L, -1, 1))/np.pi
		else:
			L += 1
		np.fill_diagonal(L, 0)
		L = np.apply_along_axis(make_knn, 1, L)
		return L ** similarity_power



	#######################################################


