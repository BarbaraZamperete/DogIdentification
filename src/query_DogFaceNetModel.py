from models import GoogLeNet
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import joblib
import joblib
import cv2
import torch


ARTEFACTS="D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/artefacts"
PRETRAINED_MODEL = "D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/2019.07.29.dogfacenet.99.h5"


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

def query_image():
    image = cv2.imread("D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/Datasets/Images_meus_dogs/Test/guida/guida.jpg")
    resized_image = cv2.resize(image, (224,224))
    query_image=np.transpose(np.array(resized_image),(2,0,1))/255.0
    q_input=torch.tensor(query_image,dtype=torch.float).unsqueeze(0)
    # q_embedding=model(q_input.cuda()).detach().cpu().numpy()
    q_input_np = q_input.detach().cpu().numpy()
    q_input_np = np.transpose(q_input_np, (0, 2, 3, 1))
    q_embedding=model.predict(q_input_np)
    q_embedding_flat = q_embedding.flatten()
    proba = knn.predict_proba([q_embedding_flat])[0]
    for p in proba:
        print(p)
    ranked = np.argsort(proba)[::-1]
    print(ranked)
    ranked_labels = le.inverse_transform(ranked)
    ranked_labels = [label for label in ranked_labels if label != '999']
    return ranked_labels


model = tf.keras.models.load_model(PRETRAINED_MODEL, custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})
knn,le=load_artefacts()
top = query_image()
print(top)