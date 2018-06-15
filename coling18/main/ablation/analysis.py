import pandas as pd
from main.data import VAD, BE5
from framework import util

Directions=['be2vad', 'vad2be']

### Hinojosa is strong outlier! Decrease without Valence is about 50%!!!
settings=['English_ANEW_Stevenson', 'German_Schmidtke', 'Polish_Imbir', 'Spanish_Hinojosa',
			'Spanish_Redondo']



results_be=pd.DataFrame(index=settings, columns=BE5)
for s in settings:
	for c in BE5:
		result=util.get_average_result_from_df('results/be2vad/lm/{}/diff_{}.tsv'\
			.format(s,c))
		results_be.loc[s,c]=result

results_vad=pd.DataFrame(index=settings, columns=VAD)
for s in settings:
	for d in VAD:
		result=util.get_average_result_from_df('results/vad2be/lm/{}/diff_{}.tsv'\
			.format(s,d))
		results_vad.loc[s,d]=result

df=results_vad.join(results_be, how='inner')
df.loc['Average']=df.mean(axis=0)
df=df*(-1) #change sign

util.save_tsv(df, 'overview_with_hinojosa.tsv')
df.to_latex('overview_with_hinojosa.tex', float_format=util.no_zeros_formatter)

df.drop(['Spanish_Hinojosa','Average'], inplace=True)
df.loc['Average']=df.mean(axis=0)
util.save_tsv(df, 'overview.tsv')
df.to_latex('overview.tex', float_format=util.no_zeros_formatter)

