import pandas as pd
from sklearn.datasets import load_digits
from module import word2vecEmbeding as f
import os
from xml.dom import minidom

X, y = load_digits(return_X_y=True)

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
    """
    Función para extraer las respuestas del fichero xml que queramos. Posteriormente se preprocesa y se transformará en un embedding con BERT.

    :param ruta:  ruta al fichero del que extraer las respuestas
    :return: lista de frases procesadas y transformadas junto a sus respectivos índices
    """
    fichero = ruta
    ValoresRespuestas = []
    respuestas = []

    doc = minidom.parse(fichero)

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
    """
    Función que procesa los embeddings obtenidos al extraer las frases del fichero.

    :param ruta: ruta del fichero con el que trabajar
    :param cota: valor umbral de similitud
    :return: Listas de estadisticos.
    """
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
                exit()
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
    """
        Función alternativa que procesa los embeddings obtenidos al extraer las frases del fichero.

        :param ruta: ruta del fichero con el que trabajar
        :param cota: valor umbral de similitud
        :return: Listas de estadisticos.
        """
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


def runBertalgoritmoEntero(Ruta):
    """
    Función que itera en todos los valores de umbrales para obtener los estadísticos de cada una y poder pintar la matriz de confusión
    y hacer el classification report.
    :param Ruta: ruta al fichero con el que queremos trabajar

    """
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
    """
    Función complementaria para la obtención de los estadísticos usando BERT.
    :param Ruta: ruta al fichero con el que queremos trabajar.
    :param cota: valor umbral de similitud.
    :param expected: lista de los valores esperados.
    :param predicted: lista de los valores predecidos.
    :return: ambas listas ampliadas con los valores del nuevo fichero.
    """
    archivos = []

    for x in os.listdir(Ruta):
        route = Ruta + "/" + x
        archivos.append(route)
    Expected = expected
    Predicted = predicted
    for i in archivos:
        Expected, Predicted = runEstadisticosConBERTopcion2(i, cota, Expected, Predicted)

    return Expected, Predicted


def extraerRespuestasTF_IDF(ruta):
    """
        Función para extraer las respuestas del fichero xml que queramos. Posteriormente se preprocesa y se transformará en un embedding con TF_IDF.

        :param ruta:  ruta al fichero del que extraer las respuestas
        :return: lista de frases procesadas y transformadas junto a sus respectivos índices
        """
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


def runEstadisticosConTF_IDF(ruta, cota):
    """
       Función que procesa los embeddings obtenidos al extraer las frases del fichero.

       :param ruta: ruta del fichero con el que trabajar
       :param cota: valor umbral de similitud
       :return: Listas de estadisticos.
       """
    respDadas, ValoresRespuestas = extraerRespuestasTF_IDF(ruta)
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


def runTF_IDFlAlgoritmoEntero(ruta):
    """
        Función que itera en todos los valores de umbrales para obtener los estadísticos de cada una y poder pintar la matriz de confusión
        y hacer el classification report.
        :param Ruta: ruta al fichero con el que queremos trabajar

        """
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
            TP, TN, FP, FN = runEstadisticosConTF_IDF(i, cota)
            TPL += (len(TP))
            TNL += (len(TN))
            FPL += (len(FP))
            FNL += (len(FN))
        print("Con cota= " + str(cota) + " Aciertos= " + str(TPL + TNL) + " Errores= " + str(
            FPL + FNL) + " Total= " + str(FPL + FNL + TPL + TNL))
        EstadisticosDeTodoLista(TPL, TNL, FPL, FNL)


def runEstadisticosConTDIDF(ruta, cota, expected: list, predicted: list):
    """
    Función complementaria para la obtención de los estadísticos usando TF_IDF.
    :param Ruta: ruta al fichero con el que queremos trabajar.
    :param cota: valor umbral de similitud.
    :param expected: lista de los valores esperados.
    :param predicted: lista de los valores predecidos.
    :return: ambas listas ampliadas con los valores del nuevo fichero.
    """
    sentences, ValoresRespuestas = extraerRespuestasTF_IDF(ruta)
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


def runTFIDFalgoritmoEstadisticos(ruta, cota, expected, predicted):
    """
        Función complementaria para la obtención de los estadísticos usando TF_IDF.
        :param Ruta: ruta al fichero con el que queremos trabajar.
        :param cota: valor umbral de similitud.
        :param expected: lista de los valores esperados.
        :param predicted: lista de los valores predecidos.
        :return: ambas listas ampliadas con los valores del nuevo fichero.
        """
    archivos = []

    for x in os.listdir(ruta):
        route = ruta + "/" + x
        archivos.append(route)
    for i in archivos:
        expected, predicted = runEstadisticosConTDIDF(i, cota, expected, predicted)

    return expected, predicted


def preprocesFlair(frase):
    """

    :param frase: frase que queremos preprocesar usando FLAIR.
    :return: embedding de la frase.
    """
    frase = Sentence(frase)
    document_embedding.embed(frase)
    ret = frase.get_embedding()
    return ret


def extraerRespuestasFlair(ruta):
    """
        Función para extraer las respuestas del fichero xml que queramos. Posteriormente se preprocesa y se transformará en un embedding con FLAIR.

        :param ruta:  ruta al fichero del que extraer las respuestas
        :return: lista de frases procesadas y transformadas junto a sus respectivos índices
        """
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
    """
        Función complementaria para la obtención de los estadísticos usando Flair.
        :param ruta: ruta al fichero con el que queremos trabajar.
        :param cota: valor umbral de similitud.
        :param expected: lista de los valores esperados.
        :param predicted: lista de los valores predecidos.
        :return: ambas listas ampliadas con los valores del nuevo fichero.
    """
    sentences, indicesRespuestas = extraerRespuestasFlair(ruta)

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


def runFlairalgoritmoEstadisticos(ruta, cota, expected, predicted):
    """
        Función complementaria para la obtención de los estadísticos usando Flair.
        :param ruta: ruta al fichero con el que queremos trabajar.
        :param cota: valor umbral de similitud.
        :param expected: lista de los valores esperados.
        :param predicted: lista de los valores predecidos.
        :return: ambas listas ampliadas con los valores del nuevo fichero.
        """
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
    """
        Función complementaria para la obtención de los estadísticos usando TF_IDF.
        :param Ruta: ruta al fichero con el que queremos trabajar.
        :param cota: valor umbral de similitud.

        :return: listas de los estadísticos.
        """
    sentences, valoresRespuestas = extraerRespuestasFlair(ruta)
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


def runFLAIRalgoritmoEntero(ruta):
    """
    Función para iterar y obtener los estadísticos.
    :param ruta: ruta del directorio que queremos procesar.

    """
    archivos = []

    for directorio in os.listdir(ruta):
        route = ruta + "/" + directorio
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


def EstadisticosDeTodoLista(TPL, TNL, FPL, FNL):
    """
    Función complementaria para pintar los estadísticos.
    """
    print(str(TPL) + " " + str(TNL) + " " + str(FPL) + " " + str(FNL) + " ")



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def huggingFaceTransformers(Sentences):
    """
    Función para obtener embeddings usando BERT.
    :param Sentences: Frases que queremos procesar.
    :return: lista de embeddings
    """
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
    """
        Función para extraer las respuestas del fichero xml que queramos. Posteriormente se preprocesa y se transformará en un embedding con BERT usando un transformador obtenido en HuggingFace.

        :param ruta:  ruta al fichero del que extraer las respuestas
        :return: lista de frases procesadas y transformadas junto a sus respectivos índices
        """
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
    """
        Función complementaria para la obtención de los estadísticos usando BERT, usando un transformador obtenido de HuggingFace.
        :param Ruta: ruta al fichero con el que queremos trabajar.
        :param cota: valor umbral de similitud.

        :return: listas de los estadísticos.
        """
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
    """
    Función complementaria usada para escribir en un fichero en csv
    """
    df = pd.DataFrame(resultados)
    df.to_csv("prueba.csv", index=False, header=False)


def runHFalgoritmoEntero(ruta):
    """
        Función para iterar y obtener los estadísticos.
        :param ruta: ruta del directorio que queremos procesar.
        """

    archivos = []
    for x in os.listdir(ruta):
        route = ruta + "/" + x
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
    """
    Función que preprocesa frases usando BERT.
    """
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
    """
    Función para obtener las frases de un fichero csv
    """
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

def ejecutarAlgoritmoFlair():
    for i in range(11):
        cota = i * 0.1
        expected = []
        predicted = []
        expected, predicted = runFlairalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-answers",
            cota, expected, predicted)
        expected, predicted = runFlairalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-domains",
            cota, expected, predicted)
        expected, predicted = runFlairalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runFlairalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runFlairalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers",
            cota, expected, predicted)
        print("Cota: " + str(cota))
        print("Matriz de confusión")
        print(confusion_matrix(expected, predicted))
        print("Classification report")
        print(classification_report(expected, predicted))

def ejecutarAlgoritmoBert():
    for i in range(11):
        cota = i * 0.1
        expected = []
        predicted = []
        expected, predicted = runEstadisticosConBERTopcion2(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-answers",
            cota, expected, predicted)
        expected, predicted = runEstadisticosConBERTopcion2(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-domains",
            cota, expected, predicted)
        expected, predicted = runEstadisticosConBERTopcion2(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runEstadisticosConBERTopcion2(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runEstadisticosConBERTopcion2(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers",
            cota, expected, predicted)
        print("Cota: " + str(cota))
        print("Matriz de confusión")
        print(confusion_matrix(expected, predicted))
        print("Classification report")
        print(classification_report(expected, predicted))

def ejecutarAlgoritmoTF_IDF():
    for i in range(11):
        cota = i * 0.1
        expected = []
        predicted = []
        expected, predicted = runTFIDFalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-answers",
            cota, expected, predicted)
        expected, predicted = runTFIDFalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-domains",
            cota, expected, predicted)
        expected, predicted = runTFIDFalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/sciEntsBank/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runTFIDFalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-questions",
            cota, expected, predicted)
        expected, predicted = runTFIDFalgoritmoEstadisticos(
            "score-freetext-answer-master/src/main/resources/corpus/semeval2013-task7/test/2way/beetle/test-unseen-answers",
            cota, expected, predicted)
        print("Cota: "+ str(cota))
        print("Matriz de confusión")
        print(confusion_matrix(expected, predicted))
        print("Classification report")
        print(classification_report(expected, predicted))

if __name__ == "__main__":

    ejecutarAlgoritmoTF_IDF()
    #ejecutarAlgoritmoFlair()
    #ejecutarAlgoritmoBert()
    exit()

