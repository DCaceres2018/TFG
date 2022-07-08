import nltk
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from scipy.spatial import distance
from module import word2vecEmbeding as f
from module import tSNE_Embeding as tsne
import os
from xml.dom import minidom
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

X, y = load_digits(return_X_y=True)

import openai

from scipy.spatial.distance import cosine
# LIBRERIAS PARA FLAIR
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings

# init embedding
glove_embedding = WordEmbeddings('glove')
flair_embeding_foward = FlairEmbeddings('news-forward')
flair_embedding_beckward = FlairEmbeddings('news-backward')

document_embedding = DocumentPoolEmbeddings([glove_embedding, flair_embedding_beckward, flair_embeding_foward])

# BERT
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# HUGGINGFACE TRANSFORMERS
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
hf_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
import torch
import torch.nn.functional as torchFunc

import csv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def extraerRespuestasBERT(ruta):
    fichero = ruta
    ValoresRespuestas = []
    respuestas = []

    doc = minidom.parse(fichero)

    question = doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    respuestaBuena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")

    respuestas.append(respuestaBuena)

    for i in respuestasDadas:
        ValoresRespuestas.append(i.getAttribute('accuracy'))
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    respDadas = [f.pre_process(i) for i in respuestas]
    sentences_embbeding = sbert_model.encode(respDadas)

    return sentences_embbeding, ValoresRespuestas


def runEstadisticosConBERT(ruta, cota):
    respDadas, ValoresRespuestas = extraerRespuestasBERT(ruta)
    FN = []
    TP = []
    FP = []
    TN = []
    cont = 0
    buena = respDadas[0]
    for i in respDadas:
        if cont != 0:
            valor = 1 - cosine(buena, i)
            if valor < 0:
                print(valor)
            accuracy = ValoresRespuestas[cont - 1]
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

    return TP, TN, FP, FN


def runEstadisticosConBERTopcion2(ruta, cota, expected: list, predicted: list):
    respDadas, ValoresRespuestas = extraerRespuestasBERT(ruta)

    cont = 0
    buena = respDadas[0]
    for i in respDadas:
        if cont != 0:
            valor = 1 - cosine(buena, i)
            if valor < 0:
                print(valor)
            accuracy = ValoresRespuestas[cont - 1]
            if valor > cota:
                if accuracy == 'correct':
                    expected.append(1)
                    predicted.append(1)
                else:
                    expected.append(0)
                    predicted.append(1)
            else:
                if accuracy == 'correct':
                    expected.append(1)
                    predicted.append(0)
                else:
                    expected.append(0)
                    predicted.append(0)
        cont += 1

    return expected, predicted


def runBertlgoritmoEntero(Ruta):
    archivos = []

    for x in os.listdir(Ruta):
        route = Ruta + "/" + x
        archivos.append(route)
    for j in range(11):
        expected = []
        predicted = []
        cota = (j * 0.1)
        for i in archivos:
            expected, predicted = runEstadisticosConBERTopcion2(i, cota, expected, predicted)

        print(classification_report(expected, predicted))
        print(confusion_matrix(expected, predicted))


def runBertlgoritmoEstadisticos(Ruta, cota, expected, predicted):
    archivos = []

    for x in os.listdir(Ruta):
        route = Ruta + "/" + x
        archivos.append(route)
    Expected = expected
    Predicted = predicted
    for i in archivos:
        Expected, Predicted = runEstadisticosConBERTopcion2(i, cota, Expected, Predicted)

    return Expected, Predicted


def extraerRespuestasVEC(ruta):
    fichero = ruta
    ValoresRespuestas = []
    respuestas = []

    doc = minidom.parse(fichero)

    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    respuestaBuena = respuestaBuena.firstChild.data
    respuestasDadas = doc.getElementsByTagName("studentAnswer")

    respuestas.append(respuestaBuena)

    for respuesta in respuestasDadas:
        ValoresRespuestas.append(respuesta.getAttribute('accuracy'))
    for cont in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[cont].firstChild.data)

    respDadas = f.featureExtraction(respuestas)
    return respDadas, ValoresRespuestas


def runEstadisticosConFeature(ruta, cota):
    respDadas, ValoresRespuestas = extraerRespuestasVEC(ruta)
    FN = []
    TP = []
    FP = []
    TN = []
    cont = 0
    buena = respDadas[0]
    for respuesta in respDadas:
        if cont != 0:
            valor = f.get_cosine_similarity(buena, respuesta)
            if valor < 0:
                print(valor)
            accuracy = ValoresRespuestas[cont - 1]
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

    return TP, TN, FP, FN


def runFeaturealgoritmoEntero(ruta):
    archivos = []
    a = ruta
    for x in os.listdir(a):
        route = a + "/" + x
        archivos.append(route)

    for j in range(11):
        FNL = 0
        TPL = 0
        FPL = 0
        TNL = 0
        cota = (j * 0.1)
        for i in archivos:
            TP, TN, FP, FN = runEstadisticosConFeature(i, cota)
            TPL += (len(TP))
            TNL += (len(TN))
            FPL += (len(FP))
            FNL += (len(FN))
        print("Con cota= " + str(cota) + " Aciertos= " + str(TPL + TNL) + " Errores= " + str(
            FPL + FNL) + " Total= " + str(FPL + FNL + TPL + TNL))
        EstadisticosDeTodoLista(TPL, TNL, FPL, FNL, cota)


def runEstadisticosConTDIDF(ruta, cota, expected: list, predicted: list):
    sentences, ValoresRespuestas = extraerRespuestasVEC(ruta)
    buena = sentences[0]

    cont = 0
    for im in sentences:
        if cont != 0:
            valor = f.get_cosine_similarity(buena, im)
            # Distance2=distance.euclidean(sentences[0],sentences[i+1])
            accuracy = ValoresRespuestas[cont - 1]
            if valor > cota:
                if accuracy == 'correct':
                    expected.append(1)
                    predicted.append(1)
                else:
                    expected.append(0)
                    predicted.append(1)
            else:
                if accuracy == 'correct':
                    expected.append(1)
                    predicted.append(0)
                else:
                    expected.append(0)
                    predicted.append(0)
            cont += 1
        else:
            cont += 1

    return expected, predicted


def runTFIDFlgoritmoEstadisticos(ruta, cota, expected, predicted):
    archivos = []

    for x in os.listdir(ruta):
        route = ruta + "/" + x
        archivos.append(route)
    for i in archivos:
        expected, predicted = runEstadisticosConTDIDF(i, cota, expected, predicted)

    return expected, predicted


def preprocesFlair(i):
    frase = Sentence(i)
    document_embedding.embed(frase)
    b = frase.get_embedding()
    return b


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
    sentences = []
    for i in entrada:
        if i != "":
            b = preprocesFlair(i)
            sentences.append(b)
        else:
            sentences.append([])
    return sentences, ValoresRespuestas


def runEstadisticosConFlairopcion2(ruta, cota, expected: list, predicted: list):
    sentences, indicesRespuestas = ExtraerRespuestasFlair(ruta)

    cont = 0
    for i in range(len(sentences) - 1):
        if sentences[i + 1] != []:
            Distance = 1 - cosine(sentences[0].cpu().numpy(), sentences[i + 1].cpu().numpy())
            # Distance2=distance.euclidean(sentences[0],sentences[i+1])
            valor = Distance
            accuracy = indicesRespuestas[i]
        else:
            valor = 0
            accuracy = indicesRespuestas[i]
        if valor > cota:
            if accuracy == 'correct':
                expected.append(1)
                predicted.append(1)
            else:
                expected.append(0)
                predicted.append(1)
        else:
            if accuracy == 'correct':
                expected.append(1)
                predicted.append(0)
            else:
                expected.append(0)
                predicted.append(0)
        cont += 1

    return expected, predicted


def runFlairlgoritmoEstadisticos(ruta, cota, expected, predicted):
    archivos = []

    for x in os.listdir(ruta):
        route = ruta + "/" + x
        archivos.append(route)
    Expected = expected
    Predicted = predicted
    for i in archivos:
        Expected, Predicted = runEstadisticosConFlairopcion2(i, cota, Expected, Predicted)

    return Expected, Predicted


def runEstadisticosConFlair(ruta, cota):
    sentences, valoresRespuestas = ExtraerRespuestasFlair(ruta)
    FN = []
    TP = []
    FP = []
    TN = []
    for i in range(len(sentences) - 1):
        if sentences[i + 1] != []:
            Distance = 1 - cosine(sentences[0].cpu().numpy(), sentences[i + 1].cpu().numpy())
            # Distance2=distance.euclidean(sentences[0],sentences[i+1])
            valor = Distance
            accuracy = valoresRespuestas[i]
        else:
            valor = 0
            accuracy = valoresRespuestas[i]
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
    return TP, TN, FP, FN


def runFLAIRalgoritmoEntero(RUTA):
    archivos = []
    a = RUTA
    for directorio in os.listdir(a):
        route = a + "/" + directorio
        archivos.append(route)
    for j in range(11):
        FNL = 0
        TPL = 0
        FPL = 0
        TNL = 0
        cota = j * 0.1
        for i in archivos:
            TP, TN, FP, FN = runEstadisticosConFlair(i, cota)
            TPL += (len(TP))
            TNL += (len(TN))
            FPL += (len(FP))
            FNL += (len(FN))
        print("Con cota= " + str(cota) + " Aciertos= " + str(TPL + TNL) + " Errores= " + str(
            FPL + FNL) + " Total= " + str(FPL + FNL + TPL + TNL))
        EstadisticosDeTodoLista(TPL, TNL, FPL, FNL, cota)


def EstadisticosDeTodoLista(TPL, TNL, FPL, FNL, cota):
    print(str(TPL) + " " + str(TNL) + " " + str(FPL) + " " + str(FNL) + " ")



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def huggingFaceTransformers(Sentences):
    sentences = Sentences
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = hf_model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = torchFunc.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def ExtraerRespuestasHuggingFace(ruta):
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

    frasesPreprocesadas = [f.pre_process(i) for i in respuestas]
    sentences = []
    for frase in frasesPreprocesadas:
        if frase != "":
            sentences.append(frase)
        else:
            sentences.append("")
    return sentences, ValoresRespuestas


def runEstadisticosConHF(ruta, cota):
    sentences, ValoresRespuestas = ExtraerRespuestasHuggingFace(ruta)
    sentences = huggingFaceTransformers(sentences)
    FN = []
    TP = []
    FP = []
    TN = []
    for i in range(len(sentences) - 1):
        if sentences[i + 1] != []:
            Distance = 1 - cosine(sentences[0].cpu().numpy(), sentences[i + 1].cpu().numpy())
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
    return TP, TN, FP, FN


def escribirResultados(resultados):
    df = pd.DataFrame(resultados)
    df.to_csv("prueba.csv", index=False, header=False)


def runHFalgoritmoEntero(RUTA):
    archivos = []
    a = RUTA
    for x in os.listdir(a):
        route = a + "/" + x
        archivos.append(route)
    for j in range(11):
        FNL = 0
        TPL = 0
        FPL = 0
        TNL = 0
        cota = j * 0.1
        for i in archivos:
            TP, TN, FP, FN = runEstadisticosConHF(i, cota)
            TPL += (len(TP))
            TNL += (len(TN))
            FPL += (len(FP))
            FNL += (len(FN))
        print("Con cota= " + str(cota) + " Aciertos= " + str(TPL + TNL) + " Errores= " + str(
            FPL + FNL) + " Total= " + str(FPL + FNL + TPL + TNL))
        EstadisticosDeTodoLista(TPL, TNL, FPL, FNL, cota)


def procesamientoBert(respuestas):
    ValoresRespuestas = []
    Frases = []

    for i in respuestas:
        ValoresRespuestas.append(i[0])
        Frases.append([i[1], i[2]])
    respDadas = [[f.pre_process(i), f.pre_process(j)] for [i, j] in Frases]

    for [i, j] in respDadas:
        Frases.append([sbert_model.encode(i), sbert_model.encode(j)])
    return Frases, ValoresRespuestas


def obtenerFrases(ruta):
    file_name = ruta
    frases = []
    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        cont = 0
        print("entro")
        for row in csv_reader:
            if cont == 0:
                cont += 1
            else:
                frases.append(row)
        print("salgo")
    return frases


if __name__ == "__main__":
    # runOpenai()
    # runTSNEopenai()
    # runTSNE()
    # runPruebaProbabilidad()
    # runPruebaProbabilidadConTruncatedSVD()
    # runEstadisticosConFlair("./sciEntsBank/test-unseen-answers/EM-inv1-45b.xml")
    # listaAc,listaErr= runFLAIRalgoritmoEntero()
    # runFLAIRalgoritmoEntero()
    for i in range(8):
        cota = i * 0.1 + 0.2
        expected = []
        predicted = []
        expected, predicted = runFlairlgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-answers",
            cota, expected, predicted)
        expected, predicted = runFlairlgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-domains",
            cota, expected, predicted)
        expected, predicted = runFlairlgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runFlairlgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runFlairlgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers",
            cota, expected, predicted)
        print(cota)
        print(confusion_matrix(expected, predicted))
        print(classification_report(expected, predicted))
    exit()
    """print("ANSWERS")
    runFeaturealgoritmoEntero("score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers")
    print("QUESTIONS")
    runFeaturealgoritmoEntero("score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions")"""
    # print("ANSWERS")
    # runFeaturealgoritmoEntero("score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-questions")
    # runFLAIRalgoritmoEntero("score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers")
    # print("QUESTIONS")
    # runFLAIRalgoritmoEntero("score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions")

    print("QUESTIONS SCB")
    runHFalgoritmoEntero(
        "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-questions")
    print("answers SCB")
    runHFalgoritmoEntero(
        "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-answers")
    print("Domains SCB")
    runHFalgoritmoEntero(
        "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-domains")
    print("QUESTIONS BT")
    runHFalgoritmoEntero(
        "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions")
    print("ANSWERS BT")
    runHFalgoritmoEntero(
        "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers")
