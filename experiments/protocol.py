import experiments as exp
from significance_tests import *
from inter_study_reliability import compute_inter_study_reliability
from resource_construction import create_ratings, count_additional_entries


all_languages=['English','Spanish', 'Polish', 'German']

### General Descriptive Statistics
def general_statistics():
	compute_inter_study_reliability(output_tex='../analysis/inter_study_reliability.tex',
									output_tsv='../analysis/inter_study_reliability.tsv')
	size_of_gold_data(outpath='../analysis/size_of_gold_data.tsv')



# ### experiments
def experiment_1():
	exp.experiment1_monolingual(output_mean_tex='../results/experiment_1_mean.tex',
								output_mean_tsv='../results/experiment_1_mean.tsv',
								output_std_tex='../results/experiment_1_std.tex',
								output_std_tsv='../results/experiment_1_std.tsv',
								languages=all_languages,
								model='knn')

def experiment_2():
	exp.experiment2_crosslingual(output_tex='../results/experiment_2.tex',
							 output_tsv='../results/experiment_2.tsv',
							 languages=all_languages,
							 model='knn')

def experiment_3():
	exp.experiment_3_exploration(languages=all_languages,
								 outpath='../analysis/experiment_3_exploration.tsv')

	exp.experiment3_all_vs_1_language(output_tex='../results/experiment_3.tex', 
								  output_tsv='../results/experiment_3.tsv',
								  train_languages=[
								  				   'English',
								  				   'Spanish',
								  				  # selection based on explorativ study
								  				  # 'German',
								  				  # 'Polish'
								  				   			 ],
								  test_languages=all_languages,
								  model='knn')


### significance tests
def significance_tests():
	significance_tests_for_experiment_1(path_inter_study_reliability='../analysis/inter_study_reliability.tsv',
										path_prediction_mean='../results/experiment_1_mean.tsv',
										path_prediction_std='../results/experiment_1_std.tsv',
										outpath_t_table='../analysis/experiment_1_t_scores.tsv',
										outpath_p_table='../analysis/experiment_1_p_values.tsv')

	significance_tests_for_experiment_2(path_inter_study_reliability='../analysis/inter_study_reliability.tsv',
										path_prediction='../results/experiment_2.tsv',
										path_gold_data_size='../analysis/size_of_gold_data.tsv',
										outpath_z_table='../analysis/experiment_2_z_scores.tsv',
										outpath_p_table='../analysis/experiment_2_p_values.tsv')

	significance_tests_for_experiment_3(path_inter_study_reliability='../analysis/inter_study_reliability.tsv',
										path_prediction='../results/experiment_3.tsv',
										path_gold_data_size='../analysis/size_of_gold_data.tsv',
										outpath_z_table='../analysis/experiment_3_z_scores.tsv',
										outpath_p_table='../analysis/experiment_3_p_values.tsv')




def resource_construction():
	# ### resource construction: Monolingual
	# Pre-load 
	english=prepare_data.get_english()
	spanish=prepare_data.get_spanish()
	polish=prepare_data.get_polish()
	german=prepare_data.get_german()

	# Warriner -> BE
	create_ratings(target_format='discrete',
				   training_df=english,
				   target_df=prepare_data.load_warriner13(),
				   out_path='../resources/EnglishBE.tsv')

	# ferre -> VAD
	create_ratings(target_format='dimensional',#
				   training_df=spanish,
				   target_df=prepare_data.load_ferre16(),
				   out_path='../resources/SpanishVAD.tsv')

	# schmidtke -> BE
	create_ratings(target_format='discrete',
				   training_df=german,
				   target_df=prepare_data.load_schmidtke14(),
				   out_path='../resources/GermanBE.tsv')


	# briesemeister -> VAD
	create_ratings(target_format='dimensional',
				   training_df=german,
				   target_df=prepare_data.load_briesemeister11(),
				   out_path='../resources/GermanVAD.tsv')

	# imbir -> BE
	create_ratings(target_format='discrete',
				   training_df=polish,
				   target_df=prepare_data.load_imbir16(),
				   out_path='../resources/PolishBE.tsv')


	### Resource Construction: Crosslingual

	#training data for crosslingual resource construction
	#  (language selection based on experiment 3 / pilot study to that.)
	full=english.append(spanish)
	print(full.shape)

	# moors -> BE
	create_ratings(target_format='discrete',
				   training_df=full,
				   target_df=prepare_data.load_moors13(),
				   out_path='../resources/DutchBE.tsv')

	# montefinese --> BE
	create_ratings(target_format='discrete',
				  training_df=full,
				  target_df=prepare_data.load_montefinese14(),
				  out_path='../resources/ItalianBE.tsv')

	# soares --> BE
	create_ratings(target_format='discrete',
				   training_df=full,
				   target_df=prepare_data.load_soares12(),
				   out_path='../resources/PortugueseBE.tsv')

	# Sianapar --> BE
	create_ratings(target_format='discrete',
				   training_df=full,
				   target_df=prepare_data.load_sianipar16(),
				   out_path='../resources/IndonesianBE.tsv')


	## Analysis of newly created lexicons
	count_additional_entries(old_df=prepare_data.load_stevenson07(),
							 new_df=pd.read_csv('../resources/EnglishBE.tsv',
							 					sep='\t',
							 					index_col=0),
							 outpath='../analysis/entries_not_in_stevenson07.txt')

	count_additional_entries(old_df=prepare_data.load_schmidtke14(),
							 new_df=pd.read_csv('../resources/GermanVAD.tsv',
							 				 sep='\t',
							 				 index_col=0),
							 outpath='../analysis/entries_not_in_schmidtke14.txt')

	count_additional_entries(old_df=prepare_data.load_briesemeister11(),
							 new_df=pd.read_csv('../resources/GermanBE.tsv',
							 					sep='\t',
							 					index_col=0),
							 outpath='../analysis/entries_not_in_briesemeister11.txt')
	count_additional_entries(old_df=prepare_data.load_wierzba15(),
							 new_df=pd.read_csv('../resources/PolishBE.tsv',
							 					sep='\t',
							 					index_col=0),
							 outpath='../analysis/entries_not_in_wierzba15.txt')

	count_additional_entries(old_df=prepare_data.load_redondo07(),
							 new_df=pd.read_csv('../resources/SpanishVAD.tsv',
							 					sep='\t',
							 					index_col=0),
							 outpath='../analysis/entries_not_in_redondo07.txt')

### execution ###
general_statistics()
experiment_1()
experiment_2()
experiment_3()
significance_tests()
resource_construction()
