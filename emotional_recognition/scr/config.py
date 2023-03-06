import torch
from torchvision import transforms


PATH='../models/affectnet_emotions/enet_b0_7.pt'

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

def loadmodel(path):
    model = torch.load(path,map_location=torch.device('cpu'))
    model=model.to(device)
    model.eval()
    return model
model = loadmodel(PATH)

IMG_SIZE=224
test_transforms = transforms.Compose(
[
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
]
)

# idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
idx_to_class={0: 'neg', 1: 'neg', 2: 'neg', 3: 'pos', 4: 'pos', 5: 'neg', 6: 'pos'}

imagespath = '../emotional_recognition/test/'