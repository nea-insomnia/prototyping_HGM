from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from scipy import ndimage
import cv2
#from tqdm import tqdm

def to_rgb(x):
    return x if x.mode == 'RGB' else x.convert('RGB')

def blur_mask(tensor):
    np_tensor = tensor.numpy()
    smoothed = ndimage.gaussian_filter(np_tensor, sigma=20)
    return torch.FloatTensor(smoothed)

def build_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), mask=False):
    #if type(image_size) != tuple:
        #image_size = (image_size, image_size)
    t = [#transforms.Resize((image_size[0], image_size[1])),
         to_rgb,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]
    if mask:
        t.append(blur_mask)
    return transforms.Compose(t)


###Load model
# Change these depending on your hardware, has to match training settings
#device = 'cuda' 
device = 'cpu'
dtype = torch.float16 


generator = torch.load("generator.pt")
generator.eval()
generator.to(device, dtype)

if device.lower() != "cpu":
    generator = generator.type(torch.half)

transform = build_transform()


###Createwebcam
#cap = cv2.VideoCapture(0)
#width, height = (480, 640)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#while True:
#    ret, frame = cap.read()
    # if not ret:
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     exit()

