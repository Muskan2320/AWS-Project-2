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

def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, epochs, gpu):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

        'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

        'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model = models.vgg16(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    classifer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, len(image_datasets['train'].classes)))]))

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = 0.001)
    
    model.classifier = classifier
    
    for e in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0

        model.train()

        for images, labels in dataloaders['train']:
            optimizer.zero_grad()

            img = model(images)
            los = loss(img, labels)
            los.backward()
            optimizer.step()

            train_loss += los.item()

        model.eval()
        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                img = model(images)
                los = loss(img, labels)
                valid_loss += los.item()

                px = torch.exp(img)
                top_p, top_class = px.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_loss /= len(dataloaders['train'])
        valid_loss /= len(dataloaders['valid'])
        accuracy /= len(dataloaders['valid'])

        print('Epoch:', e, '---------------> Training Loss:', train_loss, '  Validation Loss:', valid_loss, '  Accuracy:', accuracy*100)
        
    accuracy = 0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            img = model(images)
            px = torch.exp(img)
            top_p, top_class = px.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print('Test accuracy:', accuracy)
    
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'Model': arch,
        'Classifier': classifier,
        'class_to_idx': model.class_to_idx,
        'State_dict': model.state_dict()
    }

    torch.save(checkpoint, save_dir)
    
if __name__ == "__main__":
    args = get_input_args()
    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)