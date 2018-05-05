import pandas as pd
import scipy.stats as st
import numpy as np
import prepare_data



def ttest(xbar, mu, N, s, tails=1):
	'''
	One sample t-test from summary statistics. 
	'''
	t = (mu-xbar)*np.sqrt(float(N))/s
	if tails==1:
		if t>=0:
			p=1-st.t.cdf(x=t, df=N)
		else:
			p=1-st.t.cdf(x=-t, df=N)
	else:
		raise NotImplementedError
	return t,p

def ztest(sample_mean, pop_mean, N, pop_sd, tails=1):
	'''
	z-test
	'''
	standard_error=float(pop_sd)/np.sqrt(float(N))
	z=(float(sample_mean)-float(pop_mean))/standard_error
	if tails==1:
		if z>=0:
			p=1-st.norm.cdf(x=z)
		else:
			p=1-st.norm.cdf(x=-z)
	else:
		raise NotImplementedError
	return z,p

def correlation_test(sample_corr, pop_corr, N, tails=1):
	'''
	Tests if empirical sample correlation is significantly higher than an given 
	population correlation. Performs Fisher's r to z transformation and than 
	computes z test.
	'''
	z_r=.5*np.log((1+sample_corr)/(1-sample_corr))
	z_rho=.5*np.log((1+pop_corr)/(1-pop_corr))
	standard_error=1./(np.sqrt(float(N-3)))
	z=(z_r-z_rho)/standard_error
	if tails==1:
		if z>=0:
			p=1-st.norm.cdf(x=z)
		else:
			p=1-st.norm.cdf(x=-z)
	else:
		raise NotImplementedError
	return z,p



def size_of_gold_data(outpath):
	languages=['English', 'Spanish', 'German', 'Polish']
	data = {
		'English':prepare_data.get_english(),
		'Spanish':prepare_data.get_spanish(),
		'Polish':prepare_data.get_polish(),
		'German':prepare_data.get_german()
		}
	table=pd.DataFrame(columns=['N'], index=languages)
	for lang in languages:
		print(data[lang].shape[0])
		table['N'][lang]=data[lang].shape[0]
	table.to_csv(outpath, sep='\t')


format_string= ':02.5f'
def FORMATTER(x):
	return '{:02.5f}'.format(x)


def significance_tests_for_experiment_1(path_inter_study_reliability,
										path_prediction_mean,
										path_prediction_std,
										outpath_t_table,
										outpath_p_table):
	'''
	One-tailed one sample t-test in 10-fold cross validation setup.
 	Tests if mean over correlation values is higher than lowest inter-study
 	reliability (degrees of freedom: 9).
 	Adapted from Dietterich, T. G. (1998). Approximate statistical tests for 
 	comparing supervised classification learning algorithms. Neural 
 	Computation, 10(7), 1895–1923.

	'''
	inter_study_reliability=pd.read_csv(path_inter_study_reliability,
										index_col=0, sep='\t')

	human_floor=inter_study_reliability.min(axis=0)
	results_mean=pd.read_csv(path_prediction_mean,
							 sep='\t',
							 index_col=0)
	results_std=pd.read_csv(path_prediction_std,
							 sep='\t',
							 index_col=0)
	languages=['English', 'Spanish', 'German', 'Polish']
	dimensions=['Valence', 'Arousal', 'Dominance']

	t_table=pd.DataFrame(columns=dimensions, index=languages)
	p_table=pd.DataFrame(columns=dimensions, index=languages)

	for lang in languages:
		for dim in dimensions:
			pred=results_mean[dim][lang]
			human=human_floor[dim]
			if pred>human:
				t,p=ttest(xbar=pred,
					 	  mu=human,
					 	  N=9,
					      s=results_std[dim][lang])
				t_table[dim][lang]=t
				p_table[dim][lang]=p
	t_table=t_table.round(4)
	p_table=p_table.round(4)
	print(t_table.to_string(float_format=FORMATTER))
	print(p_table.to_string(float_format=FORMATTER))
	t_table.to_csv(outpath_t_table, sep='\t',float_format=FORMATTER)
	p_table.to_csv(outpath_p_table, sep='\t', float_format=FORMATTER)

def significance_tests_for_experiment_2(path_inter_study_reliability,
										path_gold_data_size,
										path_prediction,
										outpath_z_table,
										outpath_p_table):
	'''
	Computes significance tests whether the experimental results from
	experiment 2 are higher than the human ceiling. Uses one-tailed z-tests 
	with fisher r to z transformation.
	'''

	# Setting everything up
	languages=['English', 'Spanish', 'German', 'Polish']
	dimensions=['Valence', 'Arousal', 'Dominance']
	langcodes={'en':'English', 'es':'Spanish', 'pl':'Polish', 'de':'German'}
	inter_study_reliability=pd.read_csv(path_inter_study_reliability,
										index_col=0, sep='\t')
	gold_data_size=pd.read_csv(path_gold_data_size, sep='\t', index_col=0)
	human_floor=inter_study_reliability.min(axis=0)
	experimental_results=pd.read_csv(path_prediction, sep='\t', index_col=0)
	z_table=pd.DataFrame(columns=dimensions,
						 index=experimental_results.index)
	p_table=pd.DataFrame(columns=dimensions,
						 index=experimental_results.index)

	# Perform tests
	for case in list(experimental_results.index):
		parts=case.split('2')
		source_language=langcodes[parts[0]]
		target_language=langcodes[parts[1]]
		for dim in dimensions:
			pred=experimental_results[dim][case]
			human=human_floor[dim]
			n=gold_data_size['N'][target_language]
			# print(source_language, target_language, n, dim, pred, human)
			if pred > human:
				z,p=correlation_test(sample_corr=pred,
									 pop_corr=human,
									 N=n)
				z_table[dim][case]=z
				p_table[dim][case]=p
	# Output
	z_table=z_table.round(4)
	p_table=p_table.round(4)
	print(z_table.to_string(float_format=FORMATTER))
	z_table.to_csv(outpath_z_table, sep='\t', float_format=FORMATTER)
	print(p_table.to_string(float_format=FORMATTER))
	p_table.to_csv(outpath_p_table, sep='\t', float_format=FORMATTER)


def significance_tests_for_experiment_3(path_inter_study_reliability,
										path_gold_data_size,
										path_prediction,
										outpath_z_table,
										outpath_p_table):

	# Setting everything up
	dimensions=['Valence', 'Arousal', 'Dominance']
	inter_study_reliability=pd.read_csv(path_inter_study_reliability,
										index_col=0, sep='\t')
	gold_data_size=pd.read_csv(path_gold_data_size, sep='\t', index_col=0)
	human_floor=inter_study_reliability.min(axis=0)
	experimental_results=pd.read_csv(path_prediction, sep='\t', index_col=0)
	z_table=pd.DataFrame(columns=dimensions,
						 index=experimental_results.index)
	p_table=pd.DataFrame(columns=dimensions,
						 index=experimental_results.index)

	# Performing tests
	for lang in list(experimental_results.index):
		for dim in dimensions:
			pred=experimental_results[dim][lang]
			human=human_floor[dim]
			n=gold_data_size['N'][lang]
			if pred>human:
				z,p=correlation_test(sample_corr=pred,
					 pop_corr=human,
					 N=n)
				z_table[dim][lang]=z
				p_table[dim][lang]=p

	# Output
	z_table=z_table.round(4)
	p_table=p_table.round(4)
	print(z_table.to_string(float_format=FORMATTER))
	z_table.to_csv(outpath_z_table, sep='\t', float_format=FORMATTER)
	print(p_table.to_string(float_format=FORMATTER))
	p_table.to_csv(outpath_p_table, sep='\t', float_format=FORMATTER)


		
