import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision import transforms
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = 'cuda' if use_cuda else 'cpu'

from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)

# idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
idx_to_class={0: 'neg', 1: 'neg', 2: 'neg', 3: 'pos', 4: 'pos', 5: 'neg', 6: 'pos'}

def loadmodel(path):
    model = torch.load(path,map_location=torch.device('cpu'))
    model=model.to(device)
    model.eval()
    return model

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
    )

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def std_norm(input_vector):
    p = [round(i/sum(input_vector),3) for i in input_vector]
    return p


def find_emotion(fpath, model):
    frame_bgr=cv2.imread(fpath)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    bounding_boxes, points = imgProcessing.detect_faces(frame)
    points = points.T
   
    emo_name = []
    for bbox,p in zip(bounding_boxes, points):
        
        box = bbox.astype(int)
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]
        
        img_tensor = test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        
        scores = model(img_tensor.to(device))
        scores=scores[0].data.cpu().numpy()
        scores = std_norm(softmax(scores))
        if max(scores) < 0.45:
            emotions_name = idx_to_class[4]
        else:
            emotions_name = idx_to_class[np.argmax(scores)]
        emo_name.append(emotions_name)
    
    if emo_name.count('neg')>0 and len(emo_name)<=3:
        pass
    else:
        return fpath
    
def emo_search(imagespath):
    caught = []
    for fpath in os.listdir(imagespath)[11:13]:
        f1 = imagespath+ fpath 
        a = find_emotion(f1,model)
        if (a != None):
            # print(a)
            caught.append(a)
    return caught

def main():

    IMG_SIZE=224
    PATH='../models/affectnet_emotions/enet_b0_7.pt'
    model = loadmodel(PATH)

    imagespath = '../emotional_recognition/test/'
    imageslist = emo_search(imagespath)
    return imageslist