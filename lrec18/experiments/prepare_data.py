import numpy as np
import pandas as pd
import constants as cs
from io import StringIO

heads_vad = ['Word','Valence','Arousal','Dominance']
heads_be5 = ['Word','Joy','Anger','Sadness','Fear','Disgust']

### function that rescales data
def scaleInRange(x, oldmin, oldmax, newmin,newmax):
    #linear scaling (see Köper et al. LREC 2016). 
    return ((newmax-newmin)*(x-oldmin))/(oldmax-oldmin)+newmin

#### ENGLISH

def load_anew():
	anew = pd.read_csv(cs.anew, sep = '\t')
	anew = anew[['Word','ValMn','AroMn','DomMn']]
	anew.columns = ['Word', 'Valence', 'Arousal',
				   'Dominance']
	anew.set_index('Word', inplace=True)
	return anew


def load_stevenson07():
	stevenson07=pd.read_excel(cs.stevenson07)
	stevenson07=stevenson07[['word','mean_hap','mean_ang','mean_sad',
							 'mean_fear','mean_dis']]
	stevenson07.columns=['Word', 'Joy','Anger','Sadness','Fear','Disgust']
	stevenson07.set_index('Word', inplace=True)
	return stevenson07



def load_warriner13():
	warriner13 = pd.read_csv(cs.warriner13, sep=',')
	warriner13=warriner13[['Word','V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
	warriner13.columns=heads_vad
	warriner13.set_index('Word',inplace=True)
	return warriner13



#### SPANISH
def load_redondo07():
	redondo07=pd.read_excel(cs.redondo07)
	redondo07=redondo07[['S-Word','Val-Mn-All','Aro-Mn-All','Dom-Mn-All']]
	redondo07.columns = heads_vad
	redondo07.set_index('Word', inplace=True)
	return redondo07



def load_ferre16():
	ferre16=pd.read_excel(cs.ferre16)
	ferre16=ferre16[['Spanish_Word','Hap_Mean','Ang_Mean','Sad_Mean',
					 'Fear_Mean','Disg_Mean']]
	ferre16.columns=heads_be5
	ferre16.set_index('Word', inplace=True)
	return ferre16



#### POLISH
def load_riegel15():
	riegel15=pd.read_excel(cs.riegel15)
	riegel15=riegel15[['NAWL_word','val_M_all','aro_M_all']]
	riegel15.columns=['Word','Valence','Arousal']
	riegel15.set_index('Word', inplace=True)
	return riegel15



def load_wierzba15():
	wierzba15 = pd.read_excel(cs.wierzba15)
	wierzba15=wierzba15[['NAWL_word', 'hap_M_all', 'ang_M_all', 'sad_M_all',
						 'fea_M_all', 'dis_M_all']]
	wierzba15.columns=heads_be5
	wierzba15.set_index('Word', inplace=True)
	## rescaling basic emotions

	for cat in ['Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']:
		wierzba15[cat] = [scaleInRange(x=x, oldmin=1.,
											 oldmax=7., newmin=1., newmax=5.) 
								for x in wierzba15[cat]]
	return wierzba15


def load_imbir16():
	imbir16 = pd.read_excel(cs.imbir16)
	imbir16 = imbir16[['polish word', 'Valence_M', 'arousal_M', 'dominance_M']]
	imbir16.columns=heads_vad
	imbir16.set_index('Word', inplace=True)
	return imbir16



### GERMAN

def load_schmidtke14():
	schmidtke14=pd.read_csv(cs.schmidtke14, sep='\t')
	schmidtke14=schmidtke14[['Word','Valence','Arousal','Dominance']]
	schmidtke14.columns=heads_vad
	#setting word column to lower case for compatiblity with briesemeister11
	schmidtke14['Word']=schmidtke14['Word'].str.lower()
	schmidtke14.set_index('Word', inplace=True)
	# rescaling valence
	schmidtke14.Valence = [scaleInRange(x = x, oldmin = -3.,
									   oldmax = 3., newmin = 1., newmax=9.) 
						   for x in schmidtke14.Valence]
	return schmidtke14


def load_briesemeister11():
	briesemeister11=pd.read_excel(cs.briesemeister11)
	briesemeister11=briesemeister11[['WORD_LOWER', 'HAP_MEAN', 'ANG_MEAN',
									 'SAD_MEAN', 'FEA_MEAN', 'DIS_MEAN']]
	briesemeister11.columns=heads_be5
	briesemeister11.set_index('Word', inplace=True)
	return briesemeister11



def load_hinojosa16():
	hinojosa16a=pd.read_excel(cs.hinojosa16a)
	hinojosa16a=hinojosa16a[['Word','Val_Mn', 'Ar_Mn', 'Hap_Mn', 'Ang_Mn','Sad_Mn',
							 'Fear_Mn', 'Disg_Mn']]
	hinojosa16a.columns=['Word', 'Valence', 'Arousal',
						 'Joy','Anger','Sadness','Fear','Disgust']
	hinojosa16a.set_index('Word', inplace=True)
	hinojosa16b=pd.read_excel(cs.hinojosa16b)
	hinojosa16b=hinojosa16b[['Word', 'Dom_Mn']]
	hinojosa16b.columns=['Word','Dominance']
	hinojosa16b.set_index('Word', inplace=True)
	hinojosa=hinojosa16a.join(hinojosa16b, how='inner')
	hinojosa=hinojosa[['Valence', 'Arousal', 'Dominance',
			
					   'Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']]
	return hinojosa


def load_stadthagen16():
	stadthagen16=pd.read_csv(cs.stadthagen16, encoding='cp1252')
	stadthagen16=stadthagen16[['Word', 'ValenceMean', 'ArousalMean']]
	stadthagen16.columns=['Word', 'Valence', 'Arousal']
	stadthagen16.set_index('Word', inplace=True)
	return stadthagen16

def load_kanske10():
	with open(cs.kanske10, encoding='cp1252') as f:
		kanske10=f.readlines()
	# Filtering out the relevant portion of the provided file
	kanske10=kanske10[7:1008]
	# Creating data frame from string: 
	# https://stackoverflow.com/questions/22604564/how-to-create-a-pandas-dataframe-from-string
	kanske10=pd.read_csv(StringIO(''.join(kanske10)), sep='\t')
	kanske10=kanske10[['word', 'valence_mean','arousal_mean']]
	kanske10.columns=['Word', 'Valence', 'Arousal']
	kanske10['Word']=kanske10['Word'].str.lower()
	kanske10.set_index('Word', inplace=True)
	return kanske10


def load_guasch15():
	guasch15=pd.read_excel(cs.guasch15)
	guasch15=guasch15[['Word','VAL_M', 'ARO_M']]
	guasch15.columns=['Word', 'Valence', 'Arousal']
	guasch15.set_index('Word', inplace=True)
	return guasch15

def load_moors13():
	moors13=pd.read_excel(cs.moors13, header=1)
	moors13=moors13[['Words', 'M V', 'M A', 'M P']]
	moors13.columns=heads_vad
	moors13.set_index('Word', inplace=True)
	return moors13

def load_montefinese14():
	montefinese14=pd.read_excel(cs.montefinese14, header=1)
	montefinese14=montefinese14[['Ita_Word', 'M_Val', 'M_Aro', 'M_Dom']]
	montefinese14.columns=heads_vad
	montefinese14.set_index('Word', inplace=True)
	return montefinese14

def load_soares12():
	soares12=pd.read_excel(cs.soares12, sheetname=1)
	soares12=soares12[['EP-Word', 'Val-M', 'Arou-M', 'Dom-M']]
	soares12.columns=heads_vad
	soares12.set_index('Word', inplace=True)
	return soares12

def load_sianipar16():
	sianipar16=pd.read_excel(cs.sianipar16)
	sianipar16=sianipar16[['Words (Indonesian)', 'ALL_Valence_Mean', 'ALL_Arousal_Mean', 'ALL_Dominance_Mean']]
	sianipar16.columns=heads_vad
	sianipar16.set_index('Word', inplace=True)
	return sianipar16


def get_english():
	return load_anew().join(load_stevenson07(), how='inner')

def get_spanish():
	return load_redondo07().join(load_ferre16(), how='inner')


def get_polish():
	return load_imbir16().join(load_wierzba15(), how='inner')


def get_german():
	return load_schmidtke14().join(load_briesemeister11(), how='inner')




