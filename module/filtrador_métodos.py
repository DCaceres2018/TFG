
#preprocesamiento
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string

#lematizacion
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import  word_tokenize

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
        print(palabra, " : ",lematizer.lemmatize(palabra))
        ListLematizada.append(lematizer.lemmatize(palabra))

    return ListLematizada