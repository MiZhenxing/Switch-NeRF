import torch
import imageio 
import torchvision
import cv2
import math

def load_gigapixel_data(img_path, scale=1.0):
    img = imageio.imread(img_path) / 255.0
    H, W = img.shape[0:2]

    if scale < 1.0:
        H = math.floor(scale * H)
        W = math.floor(scale * W)

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    
    return img