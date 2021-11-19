import nltk
from module import filtrador_métodos as f
import os
from xml.dom import minidom

if __name__ == '__main__':
    ruta="./sciEntsBank"
    """Para listar todos los xml
    contenido=os.listdir(ruta)
    for i in contenido:
        cont=os.listdir(ruta+"/"+i)
        for j in cont:
            print(j)"""

    doc = minidom.parse(ruta+"/test-unseen-answers/EM-inv1-45b.xml")
    question=doc.getElementsByTagName("questionText")[0]
    respuestaBuena = doc.getElementsByTagName("referenceAnswer")[0]
    buena=respuestaBuena.firstChild.data
    respuestasDadas= doc.getElementsByTagName("studentAnswer")
    respuestas=[]
    for i in range(len(respuestasDadas)):
        respuestas.append(respuestasDadas[i].firstChild.data)

    respuestas.append(buena)
    sentence= f.pre_process(buena)
    sentenceBuenaLematizada = f.lematizacion(sentence)
    respuestasPre=[]
    respuestasLem=[]
    for i in respuestas:
        respuestasPre.append(f.pre_process(i))
    for i in respuestasPre:
        respuestasLem.append(f.lematizacion(i))



    a=f.featureExtraction(respuestas)
    for i in a:
        print("Respuesta")
        print(i)
        print("Respuesta Correcta")
        print(a[-1])
        b = f.get_cosine_similarity(i, a[-1])
        print(b)


    """corpus=["A girl is styling her hair.","A girl is styling her hair."]
    list=f.featureExtraction(corpus)
    print(list)
    print()
    d=f.get_sif_feature_vectors(corpus[0],corpus[1])
    print(d)"""
