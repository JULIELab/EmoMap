
# import word_emo.framework.reference_methods.embedding_transformer
import itertools
import tensorflow as tf 
import numpy as np
import pandas as pd 
import scipy.stats as st
# from scipy.linalg import svd
from numpy.linalg import svd
import naacl.framework.util as util
from sklearn.model_selection import KFold




class Densifier():

    def __init__(self, embeddings):
        self.embeddings=embeddings
        self.d=self.embeddings.m.shape[1]
        # self.Q=np.identity(self.d) # Orthogonal transformation matrix (numpy array)
        self.P=np.zeros(shape=[self.d,1])
        self.P[0,0]=1.
        # print(self.P)
        # print('Q\n',self.Q)

        self.Qs={} #mapping from emotional variable to matrix

        self.induced_lexicon=None #pandas data frame matching seed lexicon with
                                    # all words from embeddings
        self.seed_lexicon=None

    def fit(self, seed_lexicon, binarization_threshold=.5, alpha=.7):
        tf.reset_default_graph()
        # print(seed_lexicon)
        self.seed_lexicon=seed_lexicon
        self.induced_lexicon=pd.DataFrame(  columns=self.seed_lexicon.columns,
                                            index=self.embeddings.iw)
        binarized_lexicon=self.binarize(sd_threshold=binarization_threshold)
        
        for var in list(self.induced_lexicon):
            self.Qs[var]=self.train_Q( pos=binarized_lexicon[var]['pos'],
                                neg=binarized_lexicon[var]['neg'],
                                batch_size=100,
                                optimizer='sgd',
                                orthogonalize=False,
                                alpha=alpha,
                                training_steps=3000)
            self.induced_lexicon[var]=self.embeddings.m.dot(self.Qs[var]).dot(self.P)
                ### Todo introduce rescaling and orientation checking
            # print(self.induced_lexicon)

    def predict(self, words):
        preds=self.induced_lexicon.loc[words]
        # print('Length of input, length of output: ', len(words), len(preds))
        ### If word is not in the original embeddings space, give mean of induced values instead
        means=self.induced_lexicon.mean(axis=0)
        for word in words:
            if not word in self.induced_lexicon.index:
                preds.loc[word]=means
        
        ### drop duplicated entries. FIX! (Problem is I don't know why there are 
        ### duplicated entries in the first place...)
        preds=preds[~preds.index.duplicated(keep='first')]
        ###

        ###rescaling data to fit the range of the seed lexicon
        preds=util.scale_predictions_to_seeds(preds=preds,
                                                seed_lexicon=self.seed_lexicon)
        ##########

        return preds

    def eval(self, gold_lex):
        if self.induced_lexicon is None:
            raise ValueError('Embeddings need to be transformed first! Run "fit"!')
        else:
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


    def vec(self, word):
        return self.embeddings.represent(word)

    def train_Q(self, pos, neg, alpha, batch_size=100, optimizer='sgd', orthogonalize=True, training_steps=4000):
        '''
        Takes positive and negatives seeds to learn orthogonal transformation.
        '''

        #building cartesian product of positive and negative seeds

        with tf.Graph().as_default():

            alpha=tf.constant(alpha, dtype=tf.float32)
           
            # naninpos=False
            # for p in pos:
            #     if np.isnan(p):
            #         naninpos=True
            # print('Is there a nan in pos: ', naninpos )

            # naninneg=False
            # for n in neg:
            #     if np.isnan(n):
            #         naninneg=True
            # print('Is there a nan in neg: ', naninneg )

            pairs_separate=[i for i in itertools.product(pos, neg)]

            # naninsep=False
            # for p in pairs_separate:
            #     if np.isnan(p[0]) or np.isnan(p[1]):
            #         naninsep=True
            # print('Is there a nan in pairs_separate: ', naninsep)

            # building data matrix
            ###

            ###
            #different classes
            # data_separate=np.zeros(shape=[len(pairs_separate), self.d])
            print('len data separate: ', len(pairs_separate))
            data_separate=pd.DataFrame(pairs_separate)
            # for i_pair in range(len(pairs_separate)):
            #     a,b=pairs_separate[i_pair]
            #     a=self.seed_lexicon.index[a]
            #     b=self.seed_lexicon.index[b]
            #     data_separate[i_pair]=self.vec(a)-self.vec(b)
            # # print(pairs_separate)
            # # print(data_separate, data_separate.shape)
            del pairs_separate

            #same classes
            # pairs_align=[i for i in itertools.product(pos, pos)]+[i for i in itertools.product(neg, neg)]
            print('beginning to work on aligned pairs...')
            
            # pairs_align=[tuple(x) for x in combinations(pos,pos).union(combinations(neg,neg))]
            pairs_align=combinations(pos)+combinations(neg)

            # naninal=False
            # for p in pairs_align:
            #     if len(p)<2:
            #         print(p)
            #     if np.isnan(p[0]) or np.isnan(p[1]):
            #         naninal=True
            # print('Is there a nan in pairs_algin: ', naninal)

            print('Lenght of pairs_align: ', len(pairs_align))
            data_align=pd.DataFrame(pairs_align)
            # data_align=np.zeros(shape=[len(pairs_align), self.d])
            # for i_pair in range(len(pairs_align)):
            #     a,b=pairs_align[i_pair]
            #     a=self.seed_lexicon.index[a]
            #     b=self.seed_lexicon.index[b]
            #     data_align[i_pair]=self.vec(a)-self.vec(b)
            # # print(pairs_align)
            del pairs_align



            # setting up tensorflow graph

            Q=tf.Variable(tf.random_normal(shape=[self.d, self.d], stddev=1), name='Q')
            # Q=tf.Variable(np.identity(self.d), dtype=tf.float32, name='Q')
            P=tf.constant(self.P, dtype=tf.float32) #must be column vecotr now that e_w/v are row vectors
            e_diff=tf.placeholder(tf.float32, shape=[None, self.d], name='e_diff') #e_w - e_v for w,v are from different class
            e_same=tf.placeholder(tf.float32, shape=[None, self.d], name='e_same') # e_w - e_v for w,v are from same class



            # loss function
            QxP=tf.matmul(Q,P)
            loss_separate   = -tf.reduce_sum(
                                tf.matmul(e_diff,QxP)
                                )
            loss_align      =  tf.reduce_sum(
                                tf.matmul(e_same, QxP)
                                )
            loss=(alpha*loss_separate)+((1-alpha)*loss_align)



            ### Define optimization

            if optimizer=='sgd':
                 ## CLASSICAL SGD (according to paper)
                global_step=tf.Variable(0, trainable=False)
                starter_learning_rate=5.
                learning_rate=tf.train.exponential_decay(
                            learning_rate=starter_learning_rate,
                            global_step=global_step,
                            decay_steps=1,
                            decay_rate=.99,
                            staircase=True)
                ##Passing global_step to minimize() will increment it at each step.
                
                learning_step=(
                    tf.train.GradientDescentOptimizer(learning_rate)
                    .minimize(loss,global_step=global_step)
                    )
            
            ### same with ADAM
            elif optimizer=='adam': 
                learning_rate=tf.constant(1e-3)
                learning_step=(
                    tf.train.AdamOptimizer(learning_rate)
                    .minimize(loss))
            else:
                raise NotImplementedError

            
            with tf.Session() as sess:
                init=tf.global_variables_initializer()
                sess.run(init)

                gen_separate=Batch_Gen(data=data_separate, random=True, caller=self)
                gen_align=Batch_Gen(data=data_align, random=True, caller=self)

                last_Q=Q.eval()

                for i_step in range(training_steps):
                    
                    if orthogonalize:
                        # re-orthogonalize matrix
                        u,s,v_T=svd(Q.eval())
                        new_q = u.dot(v_T.T)
                        Q.assign(new_q).eval()

                    
                    curr_separate=gen_separate.next(n=batch_size)
                    curr_align=gen_align.next(n=batch_size)
                    curr_loss, __=sess.run(   [loss, learning_step],
                                feed_dict={ 'e_diff:0':curr_separate,
                                            'e_same:0':curr_align})
                    if i_step%100==0:
                        curr_Q=Q.eval(session=sess)
                        Q_diff=np.sum(abs(last_Q-curr_Q))
                        print(i_step, curr_loss, learning_rate.eval(), Q_diff)
                        last_Q=curr_Q



                # After successful training: store matrix
                # self.Q=Q.eval(session=sess)
                # print(self.Q)



                ### Wie kann ich die beiden training objectives zusammenbringen? 
                ### Vieleicht mt Matrix multiplication?


                print('Success')
                return Q.eval()


    # def predict(self, word):
    #     # return self.P.dot(self.Q.dot(self.embeddings.represent(word)))
    #     # return (self.Q.dot(self.P)).dot(np.reshape(self.vec(word), [1,self.d]))
    #     return np.reshape(self.vec(word), [1,self.d]).dot(self.Q.dot(self.P))[0] 


    def binarize(self, sd_threshold):
        '''
        ARGS:
        lexicon         Pandas Data Frame.
        sd_threshold    The fraction of the standard deviation above and below
                        the mean which gives the binarization threshold.

        RETURNS
        Dictionary of dictionary containing the indices (referring to self.
        seed_lexicon)
        '''

        lexicon=self.seed_lexicon
        
        means=lexicon.mean(axis=0)
        sds=lexicon.std(axis=0)
        # print(means, sds)
        binarized={var:{'pos':[], 'neg':[]} for var in list(lexicon)}
        # print(binarized)
        for i_word in range(len(lexicon)):
            #word=lexicon.index[i]
            for i_var in range(len(list(lexicon))):
                var=list(lexicon)[i_var]
                mean=means.iloc[i_var]
                sd=sds.iloc[i_var]
                # print(var,word)
                # print(word, var, mean, sd_threshold, sd)
                if lexicon.iloc[i_word,i_var]> (mean + sd_threshold*sd):
                    binarized[var]['pos']+=[i_word]
                elif lexicon.iloc[i_word,i_var]< (mean - sd_threshold*sd):
                    binarized[var]['neg']+=[i_word]
        # print('Variable, #positive seeds, #negative seeds: ',var,\
            # len(binarized[var]['pos']), len(binarized[var]['neg']))
        return binarized

# def combinations(list1, list2):
#     '''Returns all possible combinations of entries of two lists
#     (permutations minus the ones having the same elements)
#     '''
#     return {frozenset({x,y}) for x in list1 for y in list2}
    
def combinations(it):
    out=[]
    for j in range(len(it)):
        for i in range(len(it)):
            if j>i:
                out+=[(it[i],it[j])]
    return out


class Batch_Gen():
    
    def __init__(self, data, caller, random=False):
        self.data=pd.DataFrame(data)
        self.index=0
        self.random=random
        self.len=self.data.shape[0]
        self.caller=caller
        # if random

    def next(self, n):
        # out=np.zeros(shape=[n,self.data.shape[1]])
        # if not self.random:
        #     end_index = self.index+n
        #     if end_index > self.len:
        #         # batch extends over end of data, restart from beginning
        #         partial=self.len-self.index
        #         out[0:partial]=self.data[self.index:] 
        #         end_index=end_index-1
        #         out[partial:end_index]=self.data[end_index:]
        #         self.index=end_index
        #         return out
        #     else:
        #         #just fill the array
        #         out=self.data[self.index:end_index]
        #         self.index=end_index
        #         return out
        if self.random:
            pairs = self.data.sample(n=n, axis=0, replace=True)
            batch=np.zeros([len(pairs), self.caller.d])
            for i in range(len(pairs)):
                # print(pairs)
                word1=self.caller.seed_lexicon.index[pairs.iloc[i,0]]
                word2=self.caller.seed_lexicon.index[pairs.iloc[i,1]]
                batch[i]=self.caller.vec(word1)-self.caller.vec(word2)
            return batch
        else:
            raise NotImplementedError
