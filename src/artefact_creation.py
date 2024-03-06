import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from models import GoogLeNet

from loss_fn import TripletLoss
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib

from tensorflow.keras import backend as K
import tensorflow as tf


PRETRAINED_MODEL="D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/2019.07.29.dogfacenet.99.h5"
INDEXING_PATH="D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/Imagens_Avaliacao/*/*.jpg"
OUTPUT_PATH="D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/artefacts"

class DogIndexingDataset(Dataset):
    def __init__(self,paths,labels):
        self.paths=paths
        self.labels=labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        anchor_file=self.paths[idx]
        anchor_label=self.labels[idx]


        anchors=np.array(Image.open(anchor_file))

        anchors=np.transpose(anchors,(2,0,1)) /255.0

        return {"images":torch.tensor(anchors,dtype=torch.float),
                "label":anchor_label}


def gen_index_labels():
    indexed_files=glob(INDEXING_PATH)
    label_index=[t.split("\\")[-1].split(".")[0] for t in indexed_files]
    le=LabelEncoder()
    label_=le.fit_transform(label_index)
    return indexed_files,label_index,le,label_

# FUnção de perda

# Loss definition.

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

#############

def load_model():
    model=GoogLeNet()
    model.load_state_dict(torch.load(PRETRAINED_MODEL))
    return model

# model=load_model()
model = tf.keras.models.load_model(PRETRAINED_MODEL, custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})
# model.cuda()
# model.to('cpu')
# model.eval()
indexed_files,label_index,le,label_=gen_index_labels()

print(indexed_files)
print("###########")
print(label_index)
print("###########")
print(le)
print("###########")
print(label_)

ds=DogIndexingDataset(indexed_files,label_index)
dataloader = DataLoader(ds, batch_size=2,
                            shuffle=False, num_workers=0)
embeddings=[]
labels=[]
for bs in dataloader:
    labels.extend(bs["label"])
    # embeddings.extend(model(bs["images"].cuda()).detach().cpu().numpy())
    embeddings.extend(model(bs["images"]).detach().numpy())

knn=KNeighborsClassifier(3)
knn.fit(np.array(embeddings),np.array(label_))
os.makedirs(OUTPUT_PATH,exist_ok=True)
joblib.dump(knn,os.path.join(OUTPUT_PATH,"knn.pkl"))
joblib.dump(le,os.path.join(OUTPUT_PATH,"lEncoder.pkl"))