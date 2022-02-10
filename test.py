import nltk
import numpy as np
from sklearn.datasets import load_digits

from module import word2vecEmbeding as f

from module import tSNE_Embeding as tsne
import os
from xml.dom import minidom
import matplotlib.pyplot as plt

def pruebas2():

    list=['a','b','c']
    a=list.pop()
    print(a)

def pruebasWord2Bec():
    ruta = "./sciEntsBank"


    doc = minidom.parse(ruta + "/test-unseen-answers/EM-inv1-45b.xml")
    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    a=[]
    for i in respuestasDadas:
        a.append(i.getAttribute('accuracy'))
    respuestas = []

    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    print(a)



def runWord2Vec(num : int=0.4):
    cota=num
    listaNumeros=[]
    ValoresRespuestas=[]
    ruta = "./sciEntsBank"

    contenido = os.listdir(ruta)

    for i in contenido:
        directorio=ruta + "/" + i
        cont = os.listdir(directorio)
        for j in cont:
            fichero = directorio+ "/" + j

            doc = minidom.parse(fichero)

            question = doc.getElementsByTagName("questionText")[0]
            respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
            buena = respuestaBuena.firstChild.data
            respuestasDadas = doc.getElementsByTagName("studentAnswer")
            respuestas = []

            for i in respuestasDadas:
                ValoresRespuestas.append(i.getAttribute('accuracy'))
            ValoresRespuestas.append('base')
            for i in range(len(respuestasDadas)):
                respuestas.append(respuestasDadas[i].firstChild.data)
            respuestas.append(buena)


            respuestasPre = []
            respuestasLem = []
            for i in respuestas:
                respuestasPre.append(f.pre_process(i))
            for i in respuestasPre:
                respuestasLem.append(f.lematizacion(i))
            respDadas=[]
            respDadas = f.featureExtraction(respuestas)
            print(respDadas)
            for i in respDadas:
                    b = f.get_cosine_similarity(i, respDadas[-1])
                    listaNumeros.append(b)
    listaAciertos=[]
    listaErrores=[]
    for i in range(len(listaNumeros)):
        valor= listaNumeros[i]
        accuracy= ValoresRespuestas[i]
        if accuracy != 'base':
            if valor < cota:
                if accuracy == 'correct':
                    listaErrores.append((valor,accuracy))
                else:
                    listaAciertos.append((valor,accuracy))
            else:
                if accuracy == 'incorrect':
                    listaErrores.append((valor,accuracy))
                else:
                    listaAciertos.append((valor,accuracy))


    return len(listaAciertos)

def runTSNE(x):
    listaNumeros = []
    ValoresRespuestas = []
    fichero = "./sciEntsBank/test-unseen-answers/EM-inv1-45b.xml"

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []

    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    respuestas.append(buena)

    respuestasPre = []
    respuestasLem = []
    for i in respuestas:
        respuestasPre.append(f.pre_process(i))
    for i in respuestasPre:
        respuestasLem.append(f.lematizacion(i))
    respDadas = []
    respDadas = f.featureExtraction(respuestas)
    print(respDadas)
    tsne.fit(respDadas)

if __name__ == '__main__':
        runTSNE(0)