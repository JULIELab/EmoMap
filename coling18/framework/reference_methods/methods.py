import numpy as np
import pandas as pd
#from socialsent.graph_construction import similarity_matrix, transition_matrix
import functools
import numpy
# from socialsent import embedding_transformer
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from scipy import sparse
#from sklearn.linear_model import LogisticRegression, Ridge
# from JWeit.socialsent fimport embedding_transformer #densifier
# import JWeit.evaluate


def dummy(embeddings, lexicons):
    pass

def bestgen(embeddings, lexicon, **kwargs):
    #@TODO put in config file!
    numOfNeighbors = 20
    """Variante von Bestgen & Vincze's (2012) Verfahren, bei dem au"""
    
       # END OF NESTED FUNCTION
    # new_entries = model.wi.keys()
    # print("Calculated new entry set")
    # expanded_lexicon = {}
    # counter = 0
    # for x in new_entries:
    #   counter += 1
    #   if counter%100 == 0:
    #       print(counter)
    #       expanded_lexicon[x] = bootstrap_word_value(lexicon, model, x, numOfNeighbors)
    targetLex = emotionLexicon.emotionLexicon()
    for target in embeddings.get_vocabulary_index():
        targetLex.add(target, __bestgen_single_word(embeddings, lexicon, target, numOfNeighbors))
    return targetLex


def __bestgen_single_word(embeddings, lexicon, word, numOfNeighbors):
    """Nested function to calculate a single word value."""
    similar_words = []
    for entry in lexicon.words():
        if entry in embeddings.get_vocabulary_index(): #was ist denn wenn nicht? 
            similar_words.append((entry,embeddings.similarity(word,entry)))
        #else: print("Nicht im Model: ", entry)
    #print("Number of similar words: ", len(similar_words))
    similar_words.sort(key=lambda tup: tup[1])
    vad = np.array([.0,.0,.0])
    for i in range(len(similar_words)-numOfNeighbors, len(similar_words)): 
        # added check if enough similar words are there
        vad += lexicon.get(similar_words[i][0])
        # vad = add_vec(affective_value, lexicon[similar_words[i][0]])
    vad *= 1./numOfNeighbors
    return vad




def turney(embeddings, lexicon, alpha=1, threshold=0, **kwargs):
    """
    Modification of Algorithm presented in Turney and Littman (2002) where numerical seed
    values are expeted instead of positive/negative paradigm words.
    To compensate for the negative effect of taking too many seed words into account,
    we introduced an alpha parameter (with which the similarity is squared). Another parameter,
    which takles the identical problem ist the threshold, which is the minimum similarity seed and target must
    have so that this seed is taken into account for the computation of the target emotion score.
    """
    targetLex = emotionLexicon.emotionLexicon()
    for word in embeddings.get_vocabulary_index():
        targetLex.add(word, __turney_single_word(embeddings, lexicon, word))
    return targetLex
    

def __turney_single_word(embeddings, lexicon, targetWord):
    vad = np.array([.0,.0,.0])
    normalization = .0
    for entry in lexicon.words():
        vad += lexicon.get(entry)*embeddings.similarity(entry,targetWord)
        normalization += embeddings.similarity(entry, targetWord)
    return vad/normalization



def sentprop_SB(embeddings, seed_lexicon, beta=0.5, neutralValue=5., **kwargs):
    '''
    seed_lexicon....pandas dataframe
    '''
    #other beta values seem to be favorable then hamiltons default value
    print("beta : " + str(beta))
    words = embeddings.iw
    M = transition_matrix(embeddings,**kwargs)
    seeds = get_seed_emotion_matrix_SB(words, seed_lexicon, neutralValue)
    # seeds=np.array(seed_lexicon)
    labels = np.ones((len(words),3))*neutralValue
    ### Run random walk
    #print(seeds,labels)
    diff = np.sum(labels)
    iterations = 0
    # This shouldnt be hard coded I guess.
    while diff>1e-6 and iterations<200:
        print(diff)
        last_labels = np.array(labels)
        labels = beta*(M.dot(labels)) + (1-beta)*seeds
        diff = np.sum(abs(last_labels-labels))
        iterations += 1
    print("Final difference / iterations:\t" + str(diff) + "\t" + str(iterations))
    #print(labels)
    # induced_lexicon = emotionLexicon.emotionLexicon()
    # for i in range(len(words)):
    #     induced_lexicon.add(words[i],labels[i,:])
    return pd.DataFrame(labels, index=words, columns=list(seed_lexicon))


def get_seed_emotion_matrix_SB(words, seed_lexicon, neutralValue, invert=False):
    # # one column per VAD dimension
    # # invert switches order of a VAD dimension (for negative seeds)

    seed_matrix = np.ones((len(words),3))*neutralValue
    for i in range(len(words)):
        #print(words[i])
        if words[i] in seed_lexicon.index:
            seed_matrix[i,:]=seed_lexicon.loc[words[i]]
    return seed_matrix


############################################################
'''
Taken from socialsent.
This should be allright, since it does not depend on the one the shape of the
input seed lexicon.

TODO: reimplement those when publishing toolkit!

# '''
def transition_matrix(embeddings, word_net=False, first_order=False, sym=False, trans=False, **kwargs):
    """
    Build a probabilistic transition matrix from word embeddings.
    """
    if word_net:
        L =  wordnet_similarity_matrix(embeddings)
    elif not first_order:
        L = similarity_matrix(embeddings, **kwargs)
    else:
        L = embeddings.m.todense().A
    if sym:
        Dinv = np.diag([1. / np.sqrt(L[i].sum()) if L[i].sum() > 0 else 0 for i in range(L.shape[0])])
        return Dinv.dot(L).dot(Dinv)
    else:
        print('foo')
        Dinv = np.diag([1. / L[i].sum() for i in range(L.shape[0])])
        L = L.dot(Dinv)
    if trans:
        return L.T
    return L

def similarity_matrix(embeddings, arccos=False, similarity_power=1, nn=25, **kwargs):
    """
    Constructs a similarity matrix from embeddings.
    nn argument controls the degree.
    """
    def make_knn(vec, nn=nn):
        vec[vec < vec[np.argsort(vec)[-nn]]] = 0
        return vec
    L = embeddings.m.dot(embeddings.m.T)
    if sparse.issparse(L):
        L = L.todense()
    if arccos:
        L = np.arccos(np.clip(-L, -1, 1))/np.pi
    else:
        L += 1
    np.fill_diagonal(L, 0)
    L = np.apply_along_axis(make_knn, 1, L)
    return L ** similarity_power




#######################################################


def sentprop_JH(embeddings, seed_lexicon, beta=0.9, **kwargs):

    def run_random_walk(M, teleport, beta, **kwargs):
        def update_seeds(r):
            r += (1 - beta) * teleport / numpy.sum(teleport)
        return run_iterative_vanilla(M * beta, numpy.ones((M.shape[1], 3)) / M.shape[1], update_seeds, **kwargs)

    words = embeddings.get_vocabulary()
    positive_seeds = get_seed_emotion_matrix_JH(words, seed_lexicon)
    negative_seeds = get_seed_emotion_matrix_JH(words, seed_lexicon, True)

    M = transition_matrix(embeddings, **kwargs)
    rpos = run_random_walk(M,  positive_seeds, beta, **kwargs)
    rneg = run_random_walk(M,  negative_seeds, beta, **kwargs)
    #return {w: rpos[i] / (rpos[i] + rneg[i]) for i, w in enumerate(words)}
    inducedLex = emotionLexicon.emotionLexicon()
    for i,w in enumerate(words):
        inducedLex.add(w,rpos[i] / (rpos[i] + rneg[i]))
    return inducedLex


def get_seed_emotion_matrix_JH(words, seed_lexicon, invert=False):
    # one column per VAD dimension
    # invert switches order of a VAD dimension (for negative seeds)

    tuples_list = []
    for field in [0, 1, 2]:
        tuples = []
        for word in words:
            if word in seed_lexicon.words():
                x = seed_lexicon.get(word)[field]
                if not invert:
                    tuples.append(x)
                else:
                    tuples.append(10 - x)
            else:
                tuples.append(0.0)
        tuples_list.append(tuples)
    seed_matrix = numpy.array(zip(*tuples_list))
    return seed_matrix


def run_iterative_vanilla(M, r, update_seeds, max_iter=50, epsilon=1e-6, **kwargs):
    for i in range(max_iter):
        last_r = numpy.array(r)
        r = numpy.dot(M, r)
        update_seeds(r)
        if numpy.abs(r - last_r).sum() < epsilon:
            break
    return r

######################################################
### SB: not sure what these guys do.

# ### META METHODS ###

# def _bootstrap_func(embeddings, positive_seeds, negative_seeds, boot_size, score_method, seed, **kwargs):
#     numpy.random.seed(seed)
#     pos_seeds = numpy.random.choice(positive_seeds, boot_size)
#     neg_seeds = numpy.random.choice(negative_seeds, boot_size)
#     polarities = score_method(embeddings, pos_seeds, neg_seeds, **kwargs)
#     return {word: score for word, score in polarities.iteritems() if
#             not word in positive_seeds and not word in negative_seeds}


# def bootstrap(embeddings, positive_seeds, negative_seeds, num_boots=10, score_method=random_walk,
#               boot_size=7, return_all=False, n_procs=15, **kwargs):
#     pool = Pool(n_procs)
#     map_func = functools.partial(_bootstrap_func, embeddings, positive_seeds, negative_seeds,
#                                  boot_size, score_method, **kwargs)
#     polarities_list = pool.map(map_func, range(num_boots))
#     if return_all:
#         return polarities_list
#     else:
#         polarities = {}
#         for word in polarities_list[0]:
#             polarities[word] = numpy.mean(
#                 [polarities_list[i][word] for i in range(num_boots)])
#         return polarities
######################################################
