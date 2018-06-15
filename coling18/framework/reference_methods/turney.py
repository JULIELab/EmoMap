import itertools
import numpy as np
import pandas as pd 
import scipy.stats as st
# from scipy.linalg import svd
from numpy.linalg import svd
import naacl.framework.util as util
from sklearn.model_selection import KFold


class Bootstrapper():
    def __init__(self, embeddings):
        self.embeddings=embeddings
        self.subembeddings=None
        self.seed_lexicon=None #Pandas dataframe
        self.defaults=None #default values for each affective variable
                            # in case we cannot infer one for whatever reason

    def fit(self, seed_lexicon):
        self.seed_lexicon=seed_lexicon
        self.defaults=self.seed_lexicon.mean(axis=0)

    def predict(self, words):
        '''
        ARGS:
        words       List of strings. The words for which emotion scores should
                    be computed.

        RETURNS:    Pandas data frame with words as index, columns are named
                    according to the seed_lexicon.

        '''
        self.subembeddings=self.embeddings.subsample(
            kept_words=list(self.seed_lexicon.index)+list(words))
        self.subembeddings.normalize()
        print('Embedding space subsampled.')
        preds=pd.DataFrame(columns=list(self.seed_lexicon))
        for word in words:
            print('Processing: {}'.format(word))
            preds.loc[word]=self.turney(word)
        preds=util.scale_predictions_to_seeds(preds, self.seed_lexicon)
        return preds


    def turney(self, word):
        """
        Modification of Algorithm presented in Turney and Littman (2002) where numerical seed
        values are expeted instead of positive/negative paradigm words.
        To compensate for the negative effect of taking too many seed words into account,
        we introduced an alpha parameter (with which the similarity is squared). Another parameter,
        which takles the identical problem ist the threshold, which is the minimum similarity seed and target must
        have so that this seed is taken into account for the computation of the target emotion score.
        """
        score=np.array([0. for __ in list(self.seed_lexicon)])
        normalization=.0

        ### vectorization
        e_word=self.subembeddings.represent(word)
        similarities=e_word.dot(self.subembeddings.m.T)
        ###

        for entry in self.seed_lexicon.index:
            #weight=self.subembeddings.similarity(entry,word)
            weight=similarities[self.subembeddings.wi[entry]]

            if np.isnan(weight) or weight<0.:
                weight=0.
            # #### debug
            # if (not entry in self.embeddings.iw) or (not word in self.embeddings.iw):
            #     print(entry,word, weight)
            # else:
            #     print('Success!', weight)
            # ####
            score+=self.seed_lexicon.loc[entry]*weight
            normalization+=weight
        score=score/normalization
        for i in range(len(score)):
            if np.isnan(score[i]):
                score[i]=self.defaults[i]
        return score

    def eval(self, gold_lex):
        return(util.eval(gold_lex, self.predict(gold_lex.index)))
   
    def crossvalidate(self, labels, k_folds=10):
        '''
        lexicon         Pandas data frame.
        '''
        
        results_df=pd.DataFrame(columns=labels.columns)
        k=0
        kf=KFold(n_splits=k_folds, shuffle=True).split(labels)
        for __, split in enumerate(kf):
            train=labels.iloc[split[0]]
            test=labels.iloc[split[1]]
            k+=1
            print(k)
            self.fit(train)
            results_df.loc[k]=self.eval(test)
            print(results_df)
        results_df=util.average_results_df(results_df)
        return results_df 

# def __turney_single_word(embeddings, lexicon, targetWord):
#     vad = np.array([.0,.0,.0])
#     normalization = .0
#     for entry in lexicon.words():
#         vad += lexicon.get(entry)*embeddings.similarity(entry,targetWord)
#         normalization += embeddings.similarity(entry, targetWord)
#     return vad/normalization