import argparse
import pandas as pd
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from PIL import Image
from collections import OrderedDict
import json

def get_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="flowers", help="Dataset directory path")
    parser.add_argument("--save_dir", type=str, default='flower_classifier_checkpoint.pth', help="Model checkpoint directory")
    parser.add_argument("--arch", default="vgg16", help="Model architecture")
    parser.add_argument("--hidden_units", type=int, default=4096, help="Number of hidden units in model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpu", default='GPU', help="Using GPU or CPU")
     
    return parser.parse_args()

def process_image(image):
    image = Image.open(image)
    
    new_size = (256, int(256 * (image.height / image.width))) if image.width < image.height else (int(256 * (image.width / image.height)), 256)
    image = image.resize(new_size)
    
    width, height = image.size
    left = (width - 224) / 2
    upper = (height - 224) / 2
    right = left + 224
    lower = upper + 224
    image = image.crop((left, upper, right, lower))
    
    np_image = np.array(image) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std
    
    processed_image = normalized_image.transpose((2, 0, 1))
    
    return processed_image

def predict(image_path, model, topk=5):
    model.eval()
    
    processed_image = process_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_tensor = torch.from_numpy(processed_image).to(device).float()
    
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():        
        output = model(image_tensor)
        probab = torch.exp(output)
    
    top_probab, top_indices = probab.topk(topk)
    top_probab = top_probab.numpy()
    top_indices = top_indices.numpy()
    
    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices[0]]
    
    return top_probab[0].tolist(), top_classes

if __name__ == "__main__":
    args = get_input_args()
    top_probs, top_classes = predict(args.image_path, args.checkpoint, args.top_k, args.gpu)

    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    class_names = [cat_to_name[class_] for class_ in top_classes]

    for prob, class_, class_name in zip(top_probs, top_classes, class_names):
        print('Class:' class_, '\nCategory:', class_name, 'Probability:', prob)