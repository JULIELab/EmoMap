import pandas as pd
from main.data import BE5, VAD, VA
from framework.prepare_data import load_anew99, load_warriner13,\
	load_stadthagen16, load_stadthagen17, load_hinojosa16,\
	load_schmidtke14, load_vo09, load_imbir16, load_riegel15,\
	load_redondo07, load_ferre16

from framework.util import compute_isr, save_tsv


data={
	'anew':load_anew99(),
	'warriner': load_warriner13(),
	'stadthagen_va':load_stadthagen16(),
	'redondo':load_redondo07(),
	'hinojosa':load_hinojosa16(),
	'ferre':load_ferre16(),
	'stadthagen_be':load_stadthagen17(),
	'schmidtke':load_schmidtke14(),
	'bawl':load_vo09(),
	'nawl':load_riegel15(),
	'imbir':load_imbir16()
	}

pairs={
	'anew':'warriner',
	'redondo':'stadthagen_va',
	'hinojosa':'stadthagen_va',
	'ferre':'stadthagen_be',
	'hinojosa':'stadthagen_be',
	'schmidtke':'bawl',
	'nawl':'imbir'
}

results={}

for first in pairs.keys():
	second=pairs[first]
	result=compute_isr(data[first], data[second])
	print(results)
	results['{}-{}'.format(first,second)]=result

df=pd.DataFrame(index=results.keys(), columns=VAD+BE5+['N'])
for key, value in results.items():
	df.loc[key]=value

print(df)

save_tsv(df, 'results.tsv')