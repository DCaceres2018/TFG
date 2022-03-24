import nltk
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from module import word2vecEmbeding as f

from module import tSNE_Embeding as tsne
import os
from xml.dom import minidom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD

X, y = load_digits(return_X_y=True)

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 10)
import openai, numpy as np


from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine
# LIBRERIAS PARA FLAIR
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings

# init embedding
glove_embedding = WordEmbeddings('glove')
flair_embeding_foward = FlairEmbeddings('news-forward')
flair_embedding_beckward = FlairEmbeddings('news-backward')

document_embedding = DocumentPoolEmbeddings([glove_embedding, flair_embedding_beckward, flair_embeding_foward])


def pruebas2():
    list = ['a', 'b', 'c']
    a = list.pop()
    print(a)


def pruebasWord2Vec():
    ruta = "./sciEntsBank"

    doc = minidom.parse(ruta + "/test-unseen-answers/EM-inv1-45b.xml")
    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    a = []
    for i in respuestasDadas:
        a.append(i.getAttribute('accuracy'))
    respuestas = []

    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    print(a)


def runWord2Vec(num: int = 0.4):
    cota = num
    listaNumeros = []
    ValoresRespuestas = []
    ruta = "./sciEntsBank"

    contenido = os.listdir(ruta)

    for i in contenido:
        directorio = ruta + "/" + i
        cont = os.listdir(directorio)
        for j in cont:
            fichero = directorio + "/" + j

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

            print(respuestas)
            respuestasPre = []
            respuestasLem = []
            for i in respuestas:
                respuestasPre.append(f.pre_process(i))
            for i in respuestasPre:
                respuestasLem.append(f.lematizacion(i))
            respDadas = []
            respDadas = f.featureExtraction(respuestas)
            print(respDadas)
            for i in respDadas:
                b = f.get_cosine_similarity(i, respDadas[-1])
                listaNumeros.append(b)
    listaAciertos = []
    listaErrores = []
    for i in range(len(listaNumeros)):
        valor = listaNumeros[i]
        accuracy = ValoresRespuestas[i]
        if accuracy != 'base':
            if valor < cota:
                if accuracy == 'correct':
                    listaErrores.append((valor, accuracy))
                else:
                    listaAciertos.append((valor, accuracy))
            else:
                if accuracy == 'incorrect':
                    listaErrores.append((valor, accuracy))
                else:
                    listaAciertos.append((valor, accuracy))

    return len(listaAciertos)


def runTSNE():
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
    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    print(respuestas)
    respDadas = []
    respDadas = f.featureExtraction(respuestas)
    emb = tsne.fit(respDadas)

    colors = ['y', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'k']
    contador = 0
    for i in respuestas:
        print(i + " " + colors[contador])
        contador += 1
    for i in range(len(emb[:, 0])):
        plt.plot(emb[:, 0][i], emb[:, 1][i], marker='o', color=colors[i % len(colors)])
    plt.show()
    # sns.scatterplot(emb[:,0],emb[:,1], hue=y,legend='full',palette=palette)


def runOpenai():
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
    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    entrada = [i for i in respuestas]

    resp = openai.Embedding.create(
        input=entrada,
        engine="text-similarity-babbage-001")
    embBuena = resp['data'][0]['embedding']
    vectorSimilitudes = []
    for i in range(len(resp['data']) - 1):
        embeddingX = resp['data'][i + 1]['embedding']
        vectorSimilitudes.append(np.dot(embBuena, embeddingX))
    for i in range(len(vectorSimilitudes)):
        print(vectorSimilitudes[i], ValoresRespuestas[i])


def runTSNEopenai():
    listaNumeros = []
    ValoresRespuestas = []
    fichero = "./sciEntsBank/test-unseen-answers/EM-inv1-45b.xml"

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    respuestas.append(buena)
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')

    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    resp = openai.Embedding.create(
        input=respuestas,
        engine="text-similarity-babbage-001")

    embeding = []
    for cont in range(len(respuestas)):
        embeding.append(resp['data'][cont]['embedding'])

    emb = tsne.tsne_fit(embeding)
    [x, y, z] = np.transpose(emb)
    list = ['y', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'k']
    contador = 0
    for i in respuestas:
        print(i + " " + list[contador])
        contador += 1
    fig = plt.figure()
    grafica = fig.add_subplot(111, projection="3d")
    grafica.scatter(x, y, z, marker='o', c=list)
    """for i in range(len(emb)):
        plt.plot(emb[i,0],emb[i,1],emb[i,2],color=list[i],marker='o')"""
    plt.show()


def runPruebaProbabilidad():
    listaNumeros = []
    ValoresRespuestas = []
    fichero = "./sciEntsBank/test-unseen-answers/EM-inv1-45b.xml"

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    respuestas.append(buena)
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')

    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    respDadas = []
    respDadas = f.featureExtraction(respuestas)
    respuestas = []
    emb = tsne.tsne_fit(respDadas)
    print(emb[0])
    [x, y, z] = np.transpose(emb)
    list = ['y', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'k', ]
    fig = plt.figure()
    grafica = fig.add_subplot(111, projection="3d")
    grafica.scatter(x, y, z, marker='o', c=list)
    """for i in range(len(emb)):
        plt.plot(emb[i,0],emb[i,1],color=list[i],marker='o')"""
    plt.show()


def runPruebaProbabilidadConTruncatedSVD():
    listaNumeros = []
    ValoresRespuestas = []
    fichero = "./sciEntsBank/test-unseen-answers/EM-inv1-45b.xml"

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    respuestas.append(buena)
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')

    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    respDadas = []
    resp = openai.Embedding.create(
        input=respuestas,
        engine="text-similarity-babbage-001")

    embeding = []
    for cont in range(len(respuestas)):
        embeding.append(resp['data'][cont]['embedding'])
    respuestas = []
    trun = TruncatedSVD(n_components=len(embeding), n_iter=7, random_state=42)
    x_tr = trun.fit(embeding)
    print(trun.explained_variance_ratio_)
    """emb = tsne.tsne_fit(respDadas)"""
    """[x, y, z] = np.transpose(x_tr)"""
    """list=['y','r','r','r','r','b','b','b','b','g','g','g','k',]
    fig = plt.figure()
    grafica = fig.add_subplot(111, projection="3d")
    grafica.scatter(x, y, z, marker='o', c=list)
    for i in range(len(emb)):
        plt.plot(emb[i,0],emb[i,1],color=list[i],marker='o')
    plt.show()"""


def runEstadisticosConDavinvi(x,cota):
    ValoresRespuestas = []
    fichero = x

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')
    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    entrada = [f.pre_process(i) for i in respuestas]

    resp = openai.Embedding.create(
        input=entrada,
        engine="text-similarity-davinci-001")
    embBuena = resp['data'][0]['embedding']
    vectorSimilitudes = []
    for i in range(len(resp['data']) - 1):
        embeddingX = resp['data'][i + 1]['embedding']
        vectorSimilitudes.append(distance.euclidean(embBuena, embeddingX))

    listaErrores = []
    listaAciertos = []

    for i in range(len(vectorSimilitudes)):
        valor = vectorSimilitudes[i]
        accuracy = ValoresRespuestas[i]

        if valor < cota:
            if accuracy == 'correct':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
        else:
            if accuracy == 'incorrect':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
    # print("aciertos=" + str(len(listaAciertos)) + " errores=" + str(len(listaErrores)))
    return listaAciertos,listaErrores


def runEstadisticosConCurie(x,cota):
    ValoresRespuestas = []
    fichero = x
    doc = minidom.parse(fichero)
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')
    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    entrada = [f.pre_process(i) for i in respuestas]
    resp = openai.Embedding.create(
        input=entrada,
        engine="text-similarity-curie-001")
    embBuena = resp['data'][0]['embedding']
    vectorSimilitudes = []
    for i in range(len(resp['data']) - 1):
        embeddingX = resp['data'][i + 1]['embedding']
        vectorSimilitudes.append(distance.euclidean(embBuena, embeddingX))

    listaErrores = []
    listaAciertos = []

    for i in range(len(vectorSimilitudes)):
        valor = vectorSimilitudes[i]
        accuracy = ValoresRespuestas[i]
        if valor < cota:
            if accuracy == 'correct':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
        else:
            if accuracy == 'incorrect':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
    # print("aciertos=" + str(len(listaAciertos)) + " errores=" + str(len(listaErrores)))
    return listaAciertos,listaErrores


def runEstadisticosConAda(x,cota):
    ValoresRespuestas = []
    fichero = x

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    ValoresRespuestas.append('base')
    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    entrada = [f.pre_process(i) for i in respuestas]

    resp = openai.Embedding.create(
        input=entrada,
        engine="text-similarity-ada-001")
    embBuena = resp['data'][0]['embedding']
    vectorSimilitudes = []
    for i in range(len(resp['data']) - 1):
        embeddingX = resp['data'][i + 1]['embedding']
        vectorSimilitudes.append(distance.euclidean(embBuena, embeddingX))

    listaErrores = []
    listaAciertos = []

    for i in range(len(vectorSimilitudes)):
        valor = vectorSimilitudes[i]
        accuracy = ValoresRespuestas[i]
        if valor < cota:
            if accuracy == 'correct':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
        else:
            if accuracy == 'incorrect':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
    # print("aciertos=" + str(len(listaAciertos)) + " errores=" + str(len(listaErrores)))
    return listaAciertos,listaErrores

def ExtraerRespuestasBabbage(ruta,nombre):
    ValoresRespuestas = []
    fichero = ruta
    doc = minidom.parse(fichero)
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []
    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)
    entrada = [f.pre_process(i) for i in respuestas]
    entrada2 = [f.lematizacion(i) for i in entrada]

    sentences = []
    for i in entrada2:
        word = ""
        for j in i:
            word = word + j
            word += " "
        sentences.append(word)

    resp = openai.Embedding.create(
        input=sentences,
        engine="text-similarity-davinci-001")

    embBuena = resp['data'][0]['embedding']
    vectorSimilitudes = []
    for i in range(len(resp['data']) - 1):
        embeddingX = resp['data'][i + 1]['embedding']
        vectorSimilitudes.append((1 - cosine(embeddingX, embBuena)))
    return vectorSimilitudes,ValoresRespuestas

def runEstadisticosConBabbagge(x,cota,nombre):
    sentences,ValoresRespuestas=ExtraerRespuestasBabbage(x,nombre)
    listaErrores = []
    listaAciertos = []

    for i in range(len(sentences)):
        valor = sentences[i]
        accuracy = ValoresRespuestas[i]
        if valor < cota:
            if accuracy == 'correct':
                listaErrores.append((valor, accuracy))
            else:
                listaAciertos.append((valor, accuracy))
        else:
            if accuracy == 'correct':
                listaAciertos.append((valor, accuracy))
            else:
                listaErrores.append((valor, accuracy))
    # print("aciertos="+str(len(listaAciertos))+" errores="+str(len(listaErrores)))
    return listaAciertos,listaErrores

def runBabbagealgoritmoEntero(nombre):
    archivos = []
    a = "./sciEntsBank/test-unseen-answers"
    for x in os.listdir(a):
        route = a + "/" + x
        archivos.append(route)

    ListaAciertosTotal=[]
    ListaErroresTotal=[]
    for j in range(11):
        numErrores = 0
        numAciertos = 0
        listaErrores = []
        listaAciertos = []
        cota = 0.7+(0.01*j)
        for i in archivos:
            listaAciertos2, listaErrores2 = runEstadisticosConBabbagge(i, cota,nombre)
            if listaErrores2 != []:
                listaErrores.append(listaErrores2)
                numErrores += len(listaErrores2)
            if listaAciertos2 != []:
                listaAciertos.append(listaAciertos2)
                numAciertos += len(listaAciertos2)
        print("Con cota= " + str(cota) + " Aciertos= " + str(numAciertos) + " Errores= " + str(
            numErrores) + " Total= " + str(numAciertos + numErrores))
        ListaAciertosTotal.append(listaAciertos)
        ListaErroresTotal.append(listaErrores)
    return ListaAciertosTotal, ListaErroresTotal

def extraerRespuestasVEC(ruta):
    fichero = ruta
    respuestasDadad=[]
    ValoresRespuestas=[]
    sentences = []
    respuestas = []

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")

    respuestas.append(buena)

    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    respDadas = f.featureExtraction(respuestas)


    return respDadas,ValoresRespuestas

def runEstadisticosConFeature(ruta, cota):
    respDadas,ValoresRespuestas=extraerRespuestasVEC(ruta)
    FN = []
    TP = []
    FP = []
    TN = []
    cont = 0
    buena=respDadas[0]
    for i in respDadas:
        if (cont != 0):
            valor = f.get_cosine_similarity(buena, i)
            if(valor<0):
                print(valor)
            accuracy=ValoresRespuestas[cont-1]
            if valor > cota:
                if accuracy == 'correct':
                    TP.append(valor)
                else:
                    FP.append(valor)
            else:
                if accuracy == 'correct':
                    FN.append(valor)
                else:
                    TN.append(valor)
        cont += 1

    return TP,TN,FP,FN

def runFeaturealgoritmoEntero():
    archivos = []
    a = "./sciEntsBank/test-unseen-answers"
    for x in os.listdir(a):
        route = a + "/" + x
        archivos.append(route)
    ListaAciertosTotal=[]
    ListaErroresTotal=[]
    for j in range(11):
        numErrores = 0
        numAciertos = 0
        FNL = 0
        TPL = 0
        FPL = 0
        TNL= 0
        cota = (j*0.01)
        for i in archivos:
            TP,TN,FP,FN = runEstadisticosConFeature(i, cota)
            TPL+=(len(TP))
            TNL+=(len(TN))
            FPL+=(len(FP))
            FNL+=(len(FN))
        print("Con cota= " + str(cota) + " Aciertos= " + str(TPL + TNL) + " Errores= " + str(FPL + FNL) + " Total= " + str(FPL+ FNL+ TPL + TNL))
        EstadisticosDeTodoLista(TPL,TNL,FPL,FNL,cota)

def ExtraerRespuestasFlair(ruta):
    ValoresRespuestas = []
    fichero = ruta
    doc = minidom.parse(fichero)
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")
    respuestas = []

    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))

    respuestas.append(buena)
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    entrada = [f.pre_process(i) for i in respuestas]
    entrada2 = [f.lematizacion(i) for i in entrada]

    sentences = []
    for i in entrada2:
        word = ""
        for j in i:
            word = word + j
            word += " "
        frase = Sentence(word)
        document_embedding.embed(frase)
        b = frase.get_embedding()
        sentences.append(b)
    return sentences, ValoresRespuestas

def runEstadisticosConFlair(ruta, cota):
    sentences,ValoresRespuestas=ExtraerRespuestasFlair(ruta)
    FN = []
    TP = []
    FP = []
    TN = []
    for i in range(len(sentences) - 1):
        if sentences[i + 1] != []:
            Distance = 1 - cosine(sentences[0], sentences[i + 1])
            # Distance2=distance.euclidean(sentences[0],sentences[i+1])
            valor = Distance
            accuracy = ValoresRespuestas[i]
        else:
            valor = 0
            accuracy = ValoresRespuestas[i]
        if valor > cota:
            if accuracy == 'correct':
                TP.append(valor)
            else:
                FP.append(valor)
        else:
            if accuracy == 'correct':
                FN.append(valor)
            else:
                TN.append(valor)
    return TP,TN,FP,FN

def runFLAIRalgoritmoEntero():
    archivos = []
    a = "./sciEntsBank/test-unseen-answers"
    for x in os.listdir(a):
        route = a + "/" + x
        archivos.append(route)
    ListaAciertosTotal=[]
    ListaErroresTotal=[]
    for j in range(11):
        numErrores = 0
        numAciertos = 0
        FNL = 0
        TPL = 0
        FPL = 0
        TNL= 0
        cota = 0.1*j
        for i in archivos:
            TP,TN,FP,FN = runEstadisticosConFlair(i, cota)
            TPL+=(len(TP))
            TNL+=(len(TN))
            FPL+=(len(FP))
            FNL+=(len(FN))
        print("Con cota= " + str(cota) + " Aciertos= " + str(TPL + TNL) + " Errores= " + str(FPL + FNL) + " Total= " + str(FPL+ FNL+ TPL + TNL))
        EstadisticosDeTodoLista(TPL,TNL,FPL,FNL,cota)

def EstadisticosDeTodoLista(TPL,TNL,FPL,FNL,cota):
        FN= FNL
        TP= TPL
        FP= FPL
        TN= TNL
        print(str(TP)+" "+str(TN)+" "+str(FP)+" "+str(FN)+" ")
        totalEncontrados=0
        totalTodos=0
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1=(recall+precision)/2
        Accuracy=(TP+TN)/(TP+TN+FP+FN)
        print("Con cota= "+str(round(cota,2))+" .Precision= "+str(round(precision,3))+" Recall= "+str(round(recall,3))+" F1= "+str(round(f1,3))+" Accuracy= "+str(round(Accuracy,3)))



if __name__ == '__main__':
    # runOpenai()
    # runTSNEopenai()
    # runTSNE()
    # runPruebaProbabilidad()
    # runPruebaProbabilidadConTruncatedSVD()
    # runEstadisticosConFlair("./sciEntsBank/test-unseen-answers/EM-inv1-45b.xml")
    # listaAc,listaErr= runFLAIRalgoritmoEntero()
    #runFLAIRalgoritmoEntero()
    runFeaturealgoritmoEntero()
