import nltk
from module import filtrador_métodos as f
import os
from xml.dom import minidom
import matplotlib.pyplot as plt

def pruebas2():

    list=['a','b','c']
    a=list.pop()
    print(a)

def pruebas():
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



def run(num : int=0.4):
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

if __name__ == '__main__':
    aciertos= []
    for i in range(10):
        cota=0.1 * i
        aciertos.append(run(cota))
    print(aciertos)