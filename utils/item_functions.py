from __future__ import division

from itertools import combinations
from collections import defaultdict
import numpy as np
import math

from scipy import sparse, dot, linalg
from scipy.spatial.distance import cosine, cdist
from scipy import sparse
from scipy import spatial


from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity 

from gensim.models import KeyedVectors
import pandas as pd    



def import_test ():
    return "item imported"

def load_vectors (vectorsFile):
    vectorModel = KeyedVectors.load_word2vec_format(vectorsFile)
    return vectorModel

def cos_csim(matrix, vector):
    """
    Compute the cosine similarity between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    return 1 - cdist(matrix, v, 'cosine').reshape(-1)

def writeOutput(df, outfname, cosine_threshold):
    
    full_df = pd.DataFrame(columns = ['emotion', 'word', 'cosine'])

    for emotion in df.columns[1:].values:
        emotive_df = df[['word',emotion]]
        emotive_df.insert(0,'emotion',[emotion for i in range (emotive_df.shape[0])])
        emotive_df = emotive_df.rename(columns = {emotion:'cosine'})
        
        emotive_df = emotive_df[emotive_df['cosine'] >= cosine_threshold]
        emotive_df = emotive_df.sort_values(by=['cosine'],ascending=False)
        
        
        full_df = full_df.append(emotive_df,ignore_index=True)
    
    full_df.to_csv(outfname, sep = "\t", index = False)

    
    return outfname

def load_seeds(seeds_file):

    centroids_dict = dict()
    with open(seeds_file, "r") as centroids_words_file:
        next(centroids_words_file)
        for line in centroids_words_file:
            splitted_line = line.strip().split("\t")
            if len(splitted_line) == 3:
                seed, target_centroid, pos = splitted_line
                lemma_pos = "-".join([seed,pos.lower()])
                cWord = lemma_pos
            else:
                seed, target_centroid = splitted_line
                cWord = seed 
            try:
                centroids_dict[target_centroid].add(cWord)
            except KeyError:
                centroids_dict[target_centroid] = set()
                centroids_dict[target_centroid].add(cWord)
    return centroids_dict


def get_centroid_vectors(seedsDict, vectorModel):


    centroidVectors = []
    centroidIdxs = []
    centroidNames = []

    idx = 0
    for centroid_name, words in seedsDict.items():
        wordVectors = []
        centroidIdxs.append(idx)
        centroidNames.append(centroid_name)
        for word in words:
            try:
                wordVector = vectorModel[word]
                wordVectors.append(wordVector)
            except KeyError:
                continue
        #print(f' added {len(wordVectors)} words for centroid {centroid_name}')
        centroidVector = np.nanmean(wordVectors, axis = 0, dtype=float)
        centroidVectors.append(centroidVector)
        idx += 1
        
    return centroidIdxs, centroidNames, centroidVectors

def get_similatity_matrix(vectorModel,centroidVectors):
    wordVectors = []
    wordIdxs = []
    wordNames = []

    #centroidMatrix = sparse.csr_matrix(np.array(centroidVectors))

    if isinstance(vectorModel,dict):
        vocab = vectorModel.keys()
    else:
        vocab = vectorModel.vocab

    for idx, word in enumerate(vocab):
        wordIdxs.append(idx)
        wordNames.append(word)
        wordVectors.append(vectorModel[word])

    simMat = 1-spatial.distance.cdist(wordVectors,centroidVectors, metric= 'cosine')
    simMat.shape
    return simMat, wordVectors, wordIdxs, wordNames

