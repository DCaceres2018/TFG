import csv
import os
from xml.dom import minidom
import pickle

from scipy import spatial

from module import word2vecEmbeding as f
import sklearn.discriminant_analysis as sk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.neighbors import KNeighborsClassifier as knn


def EntrenamientoextraerRespuestasBERTSupervisado(ruta):
    fichero = ruta
    respuestas = []
    vectorIndices = []
    doc = minidom.parse(fichero)

    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")

    for i in respuestasDadas:
        if i.getAttribute('accuracy') == 'correct':
            vectorIndices.append(1)
        else:
            vectorIndices.append(0)
        respuestas.append((buena, i.firstChild.data))
    return respuestas, vectorIndices


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def run():
    KNN_model = knn(n_neighbors=7)
    respuestas = []
    archivos = []
    a = "../score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/training/2way/"
    for x in os.listdir(a):
        if x[0] != ".":
            for rx in os.listdir(a + x):
                route = a + x + "/" + rx
                archivos.append(route)
    vectoresRespuestas = []
    vectoresIndices = []
    for i in archivos:
        resp, indices = EntrenamientoextraerRespuestasBERTSupervisado(i)
        if len(resp) != len(indices):
            print("Por aqui debe haber un error")
        for x in resp:
            vectorA = sbert_model.encode(f.pre_process(x[0]))
            vectorB = sbert_model.encode(f.pre_process(x[1]))
            v = obtenerVectorResultante(vectorA, vectorB)
            print(v)
            vectoresRespuestas.append(v)
        for j in indices:
            vectoresIndices.append(j)

    KNN_model.fit(X=np.array(vectoresRespuestas), y=np.array(vectoresIndices))
    data = {
        'model': KNN_model
    }
    with open("../dataModelKNN.txt", "wb") as file:
        pickle.dump(data, file)

    archivosTest = []
    b = "../score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/"
    for x in os.listdir(b):
        if x[0] != ".":
            for rx in os.listdir(b + x):
                route = b + x + "/" + rx
                archivosTest.append(route)
    c = "../score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/"
    for x in os.listdir(c):
        if x[0] != ".":
            for rx in os.listdir(c + x):
                route = c + x + "/" + rx
                archivosTest.append(route)
    vectoresRespuestasTest = []
    vectoresIndicesTest = []
    for i in archivosTest:
        resp, indices = EntrenamientoextraerRespuestasBERTSupervisado(i)
        for x in resp:
            vectorA = sbert_model.encode(f.pre_process(x[0]))
            vectorB = sbert_model.encode(f.pre_process(x[1]))
            v = obtenerVectorResultante(vectorA, vectorB)
            vectoresRespuestasTest.append(v)
        for j in indices:
            vectoresIndicesTest.append(j)

    a = KNN_model.predict(vectoresRespuestasTest)
    print(confusion_matrix(vectoresIndicesTest, a))
    print(classification_report(vectoresIndicesTest, a))


def obtenerVectorResultante(x, y):
    correlacion = correlacion_jackknife(x, y)
    vector = [correlacion['promedio'], correlacion['se']]
    return vector


def pearsonJacknife(a, b):
    r, p = stats.pearsonr(a, b)
    return r, p


def correlacion_jackknife(x, y):
    '''
    Esta función aplica el método de Jackknife para el cálculo del coeficiente
    de correlación de Pearson.


    Parameters
    ----------
    x : 1D np.ndarray, pd.Series
        Variable X.

    y : 1D np.ndarray, pd.Series
        Variable y.

    Returns
    -------
    correlaciones: 1D np.ndarray
        Valor de correlación para cada iteración de Jackknife
    '''

    n = len(x)
    valores_jackknife = np.full(shape=n, fill_value=np.nan, dtype=float)

    for i in range(n):
        # Loop para excluir cada observación y calcular la correlación
        r = stats.pearsonr(np.delete(x, i), np.delete(y, i))[0]
        valores_jackknife[i] = r

    promedio_jackknife = np.nanmean(valores_jackknife)
    standar_error = np.sqrt(((n - 1) / n) * \
                            np.nansum((valores_jackknife - promedio_jackknife) ** 2))
    bias = (n - 1) * (promedio_jackknife - stats.pearsonr(x, y)[0])

    resultados = {
        'valores_jackknife': valores_jackknife,
        'promedio': promedio_jackknife,
        'se': standar_error,
        'bias': bias
    }

    return resultados


from scipy import stats

if __name__ == "__main__":
    run()
