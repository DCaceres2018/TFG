
#preprocesamiento
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string

#lematizacion
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import  word_tokenize

#extraction
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

#WORD2VEC
import numpy as np
"""model2 = Word2Vec.load('GoogleNews-vectors-negative300.bin')
import gensim.models.keyedvectors as word2vec
model = word2vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
"""

from collections import Counter
import itertools

def get_cosine_similarity(feature_vec_1,feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1,-1),feature_vec_2.reshape(1,-1))[0][0]

def map_word_frequency(document):
    return Counter(itertools.chain(*document))


def get_sif_feature_vectors(sentence1, sentence2, word_emb_model=""):
    sentence1 = [token for token in sentence1.split() if token in word_emb_model.wv.vocab]
    sentence2 = [token for token in sentence2.split() if token in word_emb_model.wv.vocab]
    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 300  # size of vectore in word embeddings
    a = 0.001
    sentence_set = []
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word_emb_model.wv[word]))  # vs += sif * word_vector
        vs = np.divide(vs, sentence_length)  # weighted average
        print(vs)
        sentence_set.append(vs)
    return sentence_set

def featureExtraction(corpus):
    for c in range (len(corpus)):
        corpus[c] = pre_process(corpus[c])

    tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
    tfidf_vect.fit(corpus)

    feature= tfidf_vect.transform(corpus)
    return feature

def pre_process(corpus):
    corpus = corpus.lower()

    stopset = stopwords.words('english') + list(string.punctuation)

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])

    corpus = unidecode(corpus)

    return corpus

def lematizacion(corpus):
    """
    Descripcion:
        La lematización es el proceso de producir variables morfologicas de una raíz base del lenguaje.

    :param corpus: Frase que descomponer
    :return: tokens lematizados.
    """
    lematizer=WordNetLemmatizer()
    palabras = word_tokenize(corpus)

    ListLematizada= []

    for palabra in palabras:
        ListLematizada.append(lematizer.lemmatize(palabra))

    return ListLematizada