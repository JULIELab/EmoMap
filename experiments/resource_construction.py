import prepare_data as data
from experiments import model_collection

###### Monolingual (that is, models were trained on same language)

def create_ratings(target_format, training_df, target_df, out_path):
	'''
	target_format: 'dimensional' or 'discrete'
	'''
	model=model_collection(model_type='knn')
	model.fit(training_df)
	if target_format=='discrete':
		target_dimcats=['Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']
	elif target_format=='dimensional':
		target_dimcats=['Valence', 'Arousal', 'Dominance']
	else:
		raise NameError('Invalid argument.'+
						'"target_format" must either be "discrete"'+
						' or "dimensional".')
	pred=model.predict(target_df)[target_dimcats]
	pred=pred.round(3)
	pred.to_csv(out_path, sep='\t', float_format='%10.3f')


def count_additional_entries(old_df, new_df, outpath):
	additional=set(new_df.index).difference(set(old_df.index))
	with open(outpath,'w') as f:
		print(len(additional), file=f)



