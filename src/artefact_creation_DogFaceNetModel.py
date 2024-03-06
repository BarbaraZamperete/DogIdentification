import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import joblib
from glob import glob

OUTPUT_PATH="D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/artefacts"

PRETRAINED_MODEL = "D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/2019.07.29.dogfacenet.99.h5"
INDEXING_PATH = "D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/Datasets/Images_meus_dogs/Train/*/*.jpg"


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

def gen_index_labels():
    indexed_files=glob(INDEXING_PATH)
    label_index=[t.split("\\")[-1].split("-")[0] for t in indexed_files]
    le=LabelEncoder()
    label_=le.fit_transform(label_index)
    return indexed_files,label_index,le,label_

model = tf.keras.models.load_model(PRETRAINED_MODEL, custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})

# Inicializar o LabelEncoder
# le = LabelEncoder()

indexed_files,label_index,le,label_=gen_index_labels()

# print(indexed_files)
# print("###########")
# print(label_index)
# print("###########")
# print(le)
# print("###########")
# print(label_)

# Listas para armazenar embeddings e labels
embeddings = []
labels = []

# Iterar sobre as pastas de indivíduos
for index, image_path in enumerate(indexed_files):
    # Carregar a imagem e pré-processá-la
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalização
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de batch
    # Obter a label do indivíduo
    label = label_index[index]
    # Obter o embedding da imagem usando o modelo
    embedding = model.predict(img_array)
    # Adicionar o embedding e a label às listas
    embeddings.append(embedding)
    labels.append(label)

# Converter listas para arrays numpy
embeddings = np.array(embeddings)
labels = np.array(labels)

# print(labels)
# print(label_)

# Exibir os embeddings e labels
# for i in range(len(embeddings)):
#     print(f"Embedding da imagem {labels[i]}: {embeddings[i]}")


knn=KNeighborsClassifier(3)
flat_embeddings = [emb.flatten() for emb in embeddings]
knn.fit(flat_embeddings, np.array(label_))
os.makedirs(OUTPUT_PATH,exist_ok=True)
joblib.dump(knn,os.path.join(OUTPUT_PATH,"knn.pkl"))
joblib.dump(le,os.path.join(OUTPUT_PATH,"lEncoder.pkl"))