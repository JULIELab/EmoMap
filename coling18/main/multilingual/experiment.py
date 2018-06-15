import pandas as pd
import numpy as np

from main.data import SETTINGS, LANGUAGES, VA, BE5, MY_MODEL
import framework.models
from framework import util


INFO='''

Crosslingual experiment described in Section 4.3 (Table 5).

'''


# for comparability, we will only look at VA!

print(LANGUAGES)

DIRECTIONS={'vad2be':{'source':VA, 'target':BE5},
			'be2vad':{'source':BE5, 'target':VA}}

results={'vad2be':pd.DataFrame(columns=BE5),
		'be2vad':pd.DataFrame(columns=VA)}



for lang in LANGUAGES: #lang = test_language

	###identify setting to train on
	train_settings=[]
	test_settings=[]
	for setting in SETTINGS:
		if setting.language==lang:
			test_settings.append(setting)
		else:
			train_settings.append(setting)


	# concatenate training data
	# including slight hack to uniquely identify words between datasets
	train_lex=pd.concat([setting.load_data() for setting in train_settings])
	train_lex=[]
	for setting in train_settings:
		lex=setting.load_data()
		lex.index=[setting.name+'_'+word for word in lex.index]
		train_lex.append(lex)
	train_lex=pd.concat(train_lex)

	###test on each of the test data sets
	for setting in test_settings:
		print('test setting: ', setting.name)

		test_lex=setting.load_data()
		test_lex.index=[setting.name+'_'+word for word in test_lex.index]


		for d in list(DIRECTIONS):
			print(d)

			### now because of the interface, the source lexicon must be concatenated
			### from the source representation of train and test lexicon

			source_rep=DIRECTIONS[d]['source']
			target_rep=DIRECTIONS[d]['target']

			source_lexicon=pd.concat([train_lex[source_rep], test_lex[source_rep]])

			target_lexicon=test_lex[target_rep]
			n_in=source_lexicon.shape[1]
			n_out=target_lexicon.shape[1]


			my_model=framework.models.Mapping_Model(
				layers=[n_in]+MY_MODEL['hidden_layers']+[n_out],
				activation=MY_MODEL['activation'],
				dropout_hidden=MY_MODEL['dropout_hidden'],
				train_steps=MY_MODEL['train_steps'],
				batch_size=MY_MODEL['batch_size'],
				optimizer=MY_MODEL['optimizer'],
				source_lexicon=source_lexicon,
				verbose=1)

			my_model.fit(train_lex.index, train_lex[target_rep])
			preds=my_model.predict(target_lexicon.index)
			preds=pd.DataFrame(preds, columns=list(target_lexicon))
			result=util.eval(preds, target_lexicon)
			print(result)
			results[d].loc[setting.name]=result
			print(results[d])

results['vad2be']['Avg_BE']=results['vad2be'].mean(axis=1)
results['be2vad']['Avg_VA']=results['be2vad'].mean(axis=1)
results=pd.concat([results[key] for key in list(DIRECTIONS)], axis=1)

util.save_tsv(results, 'results.tsv')
