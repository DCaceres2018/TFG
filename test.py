import nltk
from module import filtrador_métodos as f


if __name__ == '__main__':
    sentence= "Sample of non ASCII: Ceñía. feet How to bats remove stopwords and puntuactions?"
    frase = f.pre_process(sentence)
    f.lematizacion(frase)
