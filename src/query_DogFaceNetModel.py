from models import GoogLeNet
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import joblib
import cv2
import torch
from scipy.spatial.distance import euclidean, cosine
from collections import defaultdict


ARTEFACTS="D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/artefacts"
PRETRAINED_MODEL = "D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/2019.07.29.dogfacenet.290.h5"


alpha = 0.3
def triplet(y_true,y_pred):
    
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)
    
    return K.less(ap+alpha,an)

def load_artefacts():
    knn=joblib.load(os.path.join(ARTEFACTS,"knn.pkl"))
    le=joblib.load(os.path.join(ARTEFACTS,"lEncoder.pkl"))
    return knn,le

# def query_image():
#     image = cv2.imread("D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/Datasets/Images_meus_dogs - Copia/Test/bartho/bartho.jpg")
#     resized_image = cv2.resize(image, (224,224))

#     img_array = np.array(resized_image) / 255.0  # Normalização
#     img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de batch
#     # query_image=np.transpose(np.array(resized_image),(2,0,1))/255.0
#     # q_input=torch.tensor(query_image,dtype=torch.float).unsqueeze(0)
#     # # q_embedding=model(q_input.cuda()).detach().cpu().numpy()
#     # q_input_np = q_input.detach().cpu().numpy()
#     # q_input_np = np.transpose(q_input_np, (0, 2, 3, 1))
#     q_embedding=model.predict(img_array)
#     q_embedding_flat = q_embedding.flatten()
#     proba = knn.predict_proba([q_embedding_flat])[0]
#     print(proba)
#     print(sorted(proba))
#     print(sorted(proba, reverse=True))
#     ranked = np.argsort(proba)[::-1]
#     print(ranked)
#     ranked_labels = le.inverse_transform(ranked)
#     ranked_labels = [label for label in ranked_labels if label != '999']
#     return ranked_labels


# model = tf.keras.models.load_model(PRETRAINED_MODEL, custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})
# knn,le=load_artefacts()
# top = query_image()
# print(top)



def query_image():
    image = cv2.imread("D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/Datasets/Images_meus_dogs - Copia/Test/bartho/bartho.jpg")
    resized_image = cv2.resize(image, (224,224))

    img_array = np.array(resized_image) / 255.0  # Normalização
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de batch
    q_embedding=model.predict(img_array)
    return q_embedding

model = tf.keras.models.load_model(PRETRAINED_MODEL, custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})

# Carregar os embeddings de um arquivo usando joblib.load
embeddings = joblib.load(os.path.join(ARTEFACTS, "embeddings.pkl"))

# Carregar as labels de um arquivo usando joblib.load
labels = joblib.load(os.path.join(ARTEFACTS, "labels.pkl"))

new_embedding = query_image()
# print(new_embedding[0])
distancias_euclidiana = []
distancias_cosseno = []
for e in embeddings:
    d_e = euclidean(e[0], new_embedding[0])
    d_c = cosine(e[0], new_embedding[0])
    distancias_euclidiana.append(d_e)
    distancias_cosseno.append(d_c)

# # Juntar as distâncias com as labels
# distancia_euclidiana_e_labels = list(zip(distancias_euclidiana, labels))
# distancia_cosseno_e_labels = list(zip(distancias_cosseno, labels))

# # Ordenar com base nas distâncias euclidianas
# distancia_euclidiana_e_labels.sort(key=lambda x: x[0])
# distancia_cosseno_e_labels.sort(key=lambda x: x[0])

# Separar novamente em listas de distâncias e labels
# distancias_euclidianas_ordenadas = [x[0] for x in distancia_euclidiana_e_labels]
# distancias_cosseno_ordenadas = [x[0] for x in distancia_cosseno_e_labels]
# labels_ordenadas_euclidiana = [x[1] for x in distancia_euclidiana_e_labels]
# labels_ordenadas_cosseno = [x[1] for x in distancia_cosseno_e_labels]

# print("Distâncias Euclidianas Ordenadas:")
# print(distancias_euclidianas_ordenadas)
# print("\nLabels Ordenadas Euclidiana:")
# print(labels_ordenadas_euclidiana)
# print("\nDistâncias de Cosseno Ordenadas:")
# print(distancias_cosseno_ordenadas)
# print("\nLabels Ordenadas Cosseno:")
# print(labels_ordenadas_cosseno)

# Criar dicionários para armazenar as médias das distâncias euclidianas e de cosseno para cada cachorro
media_dist_euclidiana = defaultdict(list)
media_dist_cosseno = defaultdict(list)

# Preencher os dicionários com as médias das distâncias para cada cachorro
for dist_euc, dist_cos, label in zip(distancias_euclidiana, distancias_cosseno, labels):
    media_dist_euclidiana[label].append(dist_euc)
    media_dist_cosseno[label].append(dist_cos)

# Calcular as médias das distâncias para cada cachorro
for cachorro in media_dist_euclidiana:
    media_dist_euclidiana[cachorro] = sum(media_dist_euclidiana[cachorro]) / len(media_dist_euclidiana[cachorro])

for cachorro in media_dist_cosseno:
    media_dist_cosseno[cachorro] = sum(media_dist_cosseno[cachorro]) / len(media_dist_cosseno[cachorro])

# Ordenar os cachorros com base nas médias das distâncias
cachorros_ordenados_euclidiana = sorted(media_dist_euclidiana.items(), key=lambda x: x[1])
cachorros_ordenados_cosseno = sorted(media_dist_cosseno.items(), key=lambda x: x[1])

print("Cachorros Ordenados por Média de Distância Euclidiana:")
print(cachorros_ordenados_euclidiana)
print("\nCachorros Ordenados por Média de Distância de Cosseno:")
print(cachorros_ordenados_cosseno)

joblib.dump(cachorros_ordenados_euclidiana, os.path.join(ARTEFACTS, "cachorros_ordenados_euclidiana.pkl"))
joblib.dump(cachorros_ordenados_cosseno, os.path.join(ARTEFACTS, "cachorros_ordenados_cosseno.pkl"))