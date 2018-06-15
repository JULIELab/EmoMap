import numpy as np 

import pandas as pd 
import framework.constants as cs 
import framework.prepare_data as data
import framework.util as util
import scipy.stats as st
import framework.models
from  sklearn.linear_model import LinearRegression as LM
from sklearn.neighbors import KNeighborsRegressor as KNN
from framework.reference_methods.aicyber import MLP_Ensemble

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.backend as K

INFO='''

Monolingual experiment. Results are given in Table 4 in the paper.

'''

#### ensuring reproducability
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
####




from main.data import SETTINGS, DIRECTIONS, GET_EMBEDDINGS, MY_MODEL, VLIMIT,\
	KFOLD
import main.data


 

for setting in SETTINGS:
	print(setting.name)
	embeddings=GET_EMBEDDINGS(setting.language)
	gold_lex=setting.load_data()
	for curr_dir in list(DIRECTIONS):
		source_lexicon=util.get_columns_if_existing(df=gold_lex, 
													cols=DIRECTIONS[curr_dir]['source'])
		target_lexicon=util.get_columns_if_existing(df=gold_lex,
													cols=DIRECTIONS[curr_dir]['target'])

		n_in=source_lexicon.shape[1]
		n_out=target_lexicon.shape[1]
		baseline=MLP_Ensemble(embeddings=embeddings)
		my_model=framework.models.Mapping_Model(
			layers=[n_in]+MY_MODEL['hidden_layers']+[n_out],
			activation=MY_MODEL['activation'],
			dropout_hidden=MY_MODEL['dropout_hidden'],
			train_steps=MY_MODEL['train_steps'],
			batch_size=MY_MODEL['batch_size'],
			optimizer=MY_MODEL['optimizer'],
			source_lexicon=source_lexicon)

		reference_LM=framework.models.SKlearn_Mapping_Model(base_model=LM(),
			source_lexicon=source_lexicon)
		reference_KNN=framework.models.SKlearn_Mapping_Model(base_model=KNN(n_neighbors=20),
			source_lexicon=source_lexicon)

		ev=framework.models.Evaluator(models={	'baseline':baseline,
												'Reference_KNN':reference_KNN,
												'reference_LM':reference_LM,
												'my_model':my_model})
		ev.crossvalidate(	words=gold_lex.index, 
							labels=target_lexicon, 
							k_splits=KFOLD, 
							outpath='results/{}/{}'.format(curr_dir, setting.name))





