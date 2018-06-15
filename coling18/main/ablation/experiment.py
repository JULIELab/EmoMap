import numpy as np 

import pandas as pd 
import framework.constants as cs 
import framework.prepare_data as data
import framework.util as util
import scipy.stats as st
import framework.models
from  sklearn.linear_model import LinearRegression as LM
from sklearn.neighbors import KNeighborsRegressor as KNN
import os


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.backend as K

from main.data import SETTINGS, VAD, BE5, DIRECTIONS

from framework.util import average_subdirs


INFO='''

Ablation experiment in Section 4.1 (Figure 2).

'''

k_fold=10

MODELS={#'knn':KNN(n_neighbors=20), #only use linreg because no hyperparas
		'lm':LM()}

for setting in SETTINGS:
	print(setting.name)
	gold_lex=setting.load_data()

	#this experiment will only be performed for those gold lexicons which
	#also have dominance.
	if 'Dominance' in gold_lex.columns: 
		for curr_dir in list(DIRECTIONS):
			print(curr_dir)
			source_lexicon=gold_lex[DIRECTIONS[curr_dir]['source']]
			target_lexicon=gold_lex[DIRECTIONS[curr_dir]['target']]

			for base_model_name, base_model in MODELS.items():
				print(base_model_name)
				### create an individual model each with a cropped source lexicon
				models={}
				models['full']=framework.models.SKlearn_Mapping_Model(
					base_model=base_model,
					source_lexicon=source_lexicon
					)
				for var in list(source_lexicon):
					models[var]=framework.models.SKlearn_Mapping_Model(
						base_model=base_model,
						source_lexicon=source_lexicon.drop(var, axis=1)
						)

				# Run actual evaluation
				ev=framework.models.Evaluator(models=models)
				ev.crossvalidate(words=target_lexicon.index, 
								labels=target_lexicon,
								k_splits=k_fold,
								outpath='results/{}/{}/{}/'.format(curr_dir, base_model_name,setting.name )
									 )

				### compute difference to full model:
				df_full=util.load_tsv('results/{}/{}/{}/full.tsv'.format(curr_dir, base_model_name,setting.name ))
				print(df_full)
				for var in list(source_lexicon):
					df_var=util.load_tsv('results/{}/{}/{}/{}.tsv'.format(curr_dir, base_model_name,setting.name , var))
					print(df_var)
					df_diff=df_var-df_full
					print(df_diff)
					util.save_tsv(
						df=df_diff, 
						path='results/{}/{}/{}/diff_{}.tsv'.format(curr_dir, base_model_name,setting.name , var)
						)

### compute average values
average_subdirs('results/be2vad/lm')
average_subdirs('results/vad2be/lm')