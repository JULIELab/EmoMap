import os


resources=os.environ['JULIE']+'/resources/'
# base=os.environ['JULIE']+'/'
# project_root=base+'research/coling-2018-emoMap/'
# results_path=project_root+'results/'

#==================================================#
'''
INSTRUCTIONS:
In order to run these scripts, please replace each of the paths below with the
corresponding one on your system. The paths should point at the file 
*as they were downloaded* from the respective publications. That is, without 
any further modification (i.e. deleting/renaming columns, export as csv,...). 
The script "prepare_data.py" is designed to do all these modification steps for
you.
'''

anew10 = resources+'ANEW2010/ANEW 2010/ANEW2010All.txt'
anew99=resources+'ANEW1999/ANEW1999.csv'
stevenson07 = resources+'Stevenson-BRM-2007- Emo cats for ANEW/Stevenson(2007)-ANEW_emotional_categories.xls'
redondo07=resources+'Redondo-2007-Spanish ANEW/Redondo(2007).xls'
ferre16=resources+'Ferre-2016-Spanish discrete emotion words/13428_2016_768_MOESM1_ESM.xls'
riegel15=resources+'Riegel-2015-NAWL/13428_2014_552_MOESM1_ESM.xlsx'
wierzba15 = resources+'Wierzba-2015-BEs for NAWL/journal.pone.0132305.s004.XLSX'
imbir16 = resources+'Imbir-2016-Affective_Norms_for_Polish_Words/data sheet 1.xlsx'
warriner13=resources+'Warriner/Ratings_Warriner_et_al.csv'
#schmidtke14=resources+'ANGST2014/ANGST2014.csv'
schmidtke14=resources+'ANGST2014/ratings.xlsx'
briesemeister11=resources+'Briesemeister-2011-Categorical Norms for BAWL/13428_2011_59_MOESM1_ESM.xls'
hinojosa16a=resources+'Hinojosa-2016-VA and Cats for Spanish Words/Hinojosa et al_Supplementary materials.xlsx'
hinojosa16b=resources+'Hinojosa-2016b/Hinojosa%20et%20al._Supplementary%20materials.xlsx'
stadthagen16=resources+'Stadthagen-Gonzalez-2016-Valence Arousal Norms for Spanish Words/13428_2015_700_MOESM1_ESM.csv'
stadthagen17=resources+'Stadthagen-2017-10k Discrete Norm Spanish/Stadthagen et al Spanish Discrete Emotional Norms R1.csv'
kanske10=resources+'Kanske-2010-Leipzig Affective Norms/Kanske-BRM-2010/LANG_database.txt'
vo09=resources+'Vo-2009-Berlin Affective Word List/BAWL-R.csv'
guasch15=resources+'Guasch-2015-Spanish Affective Norms/13428_2015_684_MOESM1_ESM.xlsx'
moors13=resources+'Moors-2013-Affective Norms for Dutch/13428_2012_243_MOESM1_ESM.xlsx'
montefinese14=resources+'Montefinese-2013-Affective Norms for Italian Words/Montefinese et al_Database.xls'
soares12=resources+'Soares-2012-Adaptation of ANEW for European Portuguese/13428_2011_131_MOESM1_ESM.xls'
sianipar16=resources+'Sianipar-2016-Affective Norms for Indonesian Words/data sheet 1.xlsx'
yu16=resources+'Yu-2016-Chinese Valence Arousal Words (CVAW)/v2/cvaw2.csv'

yao16=resources+'Yao-2016-Affective Norms for Chinese/yao16.txt' #Instruction: copy all text, paste into text file, insert "Chinese" as first line.
monnier14=resources+'Monnier-2014-Affective norms for french words/13428_2013_431_MOESM2_ESM.xlsx'
davidson14=resources+'Davidson-2014-Swedish affective words/davidson14.csv'
#Instruction: copy the content of Appendix B in the pdf file and creat a csv file with '\t' as delimiter, the header is:
#Ord	Antal bokstäver	Bloggmix (frekvens per miljon)	Göteborgsposten 2012 (frekvens per miljon)	Valens 	SD Valens	Arousal	SD Arousal
eilola10=resources+'Eilola-BRM-2010/Normative_ratings.xls'
engelthaler17=resources+'Engelthaler-2017-Humor norms/humor_dataset.csv'
ric13=resources+'Ric-2013-French Discrete Norms/EmoNormsDatabase.xlsx'
palogiannidi16=resources+'Palogiannidi16_greek_affective_lexicon/greek_affective_lexicon.csv'

#==================================================#

google_news_embeddings=resources+'GoogleNews-SGNS-Vectors/GoogleNews-vectors-negative300.bin'
fasttext_wikipedia_en='/data/data_buechel/fasttext_vectors/wikipedia/wiki.en.vec'

facebook_fasttext_wikipedia={
	'english':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.en.vec',
	'german':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.de.vec',
	'spanish':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.es.vec',
	'polish':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.pl.vec',
	'french':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.fr.vec',
	'italian':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.it.vec',
	'portuguese':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.pt.vec',
	'dutch':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.nl.vec',
	'swedish':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.sv.vec',
	'finnish':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.fi.vec',
	'chinese':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.zh.vec',
	'indonesian':'/data/data_buechel/fasttext_vectors/wikipedia/wiki.id.vec'
}

facebook_fasttext_common_crawl='/data/data_buechel/fasttext_vectors/common_crawl/crawl-300d-2M.vec.zip'

sedoc17_embeddings='/data/data_buechel/sedoc_eacl_2017_embeddings/embeddings_sedoc_eacl17.txt'