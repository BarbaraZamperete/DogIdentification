from models import GoogLeNet
import torch
import joblib
import os
import cv2
import numpy as np


ARTEFACTS="C:/Users/Bárbara Z/Desktop/TCC-Codes/DogIdentificationCNN/artefacts"
PRETRAINED_MODEL = "C:/Users/Bárbara Z/Desktop/TCC-Codes/DogIdentificationCNN/modelsave/best_model.pth"

def load_model():
    model=GoogLeNet()
    model.load_state_dict(torch.load(PRETRAINED_MODEL))
    return model
def load_artefacts():
    knn=joblib.load(os.path.join(ARTEFACTS,"knn.pkl"))
    le=joblib.load(os.path.join(ARTEFACTS,"lEncoder.pkl"))
    return knn,le

knn,le=load_artefacts()
model=load_model()
model.eval()

def query_image():
    image = cv2.imread("DogIdentificationCNN/dogs_256class/test_200_single_img/Adagio.jpg")
    resized_image = cv2.resize(image, (224,224))
    query_image=np.transpose(np.array(resized_image),(2,0,1))/255.0
    q_input=torch.tensor(query_image,dtype=torch.float).unsqueeze(0)
    # q_embedding=model(q_input.cuda()).detach().cpu().numpy()
    q_embedding=model(q_input).detach().numpy()
    ranked=np.argsort(knn.predict_proba(q_embedding)[0])[::-1]
    ranked=list(le.inverse_transform(ranked)[:5])
    ranked=[t for t in ranked if t!='999']
    return ranked

top = query_image()
print(top)