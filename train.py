# -*- coding: utf-8 -*-
"""
Udacity project

By Jean Desire.
"""
# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np
import torch
import os
import time
import random
import json
from torch import nn,optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms,models
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import copy
import argparse
from os.path import isdir

# =============================================================================
# Define Functions
# =============================================================================

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture to parser
    parser.add_argument('--structures', 
                        type=str, 
                        help='Choose architecture from torchvision.models')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str.')
    
    # Add hyperparameter 
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help=' learning rate')
    parser.add_argument('--fc1', 
                        type=int, 
                        help='Hidden units for DNN')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU ')
    # Add data set director
    parser.add_argument('data_dir' ,
                        action='store_true', default='./flowers/',
                       help='Data set director')
    
    # Parse args
    args = parser.parse_args()
    return args

def train_transformer(train_dir):
    # Define transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

# Function test_transformer(test_dir) performs  transformations on test and validation dataset
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=32)
    return loader

# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

def primaryloader_model(structures="alexnet"):
    # Load Defaults if none specified
    if type(structures) == type(None): 
        model = models.alexnet(pretrained=True)
        model.name = "alexnet"
        #print("Network architecture specified as vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(structures))
        model.name = structures
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

def initial_classifier(model, fc1):
    
    fc1 = 102
    # Find Input Layers
    #input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
            ("inputs", nn.Linear(9216, fc1, bias=True)),
            ("relu_1", nn.ReLU()),
            ("dropout1",nn.Dropout(p=0.5)),    
            ("fc1", nn.Linear(fc1,90)),
            ("relu_2",nn.ReLU()),
            ("fc2", nn.Linear(90,70)),
            ("relu_3", nn.ReLU()),
            ("fc3", nn.Linear(70,102)),
            ("output", nn.LogSoftmax(dim=1))
        ]))
    return classifier

def validation(model, test_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(test_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def train_model(Model, Train_loader, Test_loader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps,valid_loader):
    # Check Model Kwarg
    if type(Epochs) == type(None):
        Epochs = 10
        print("Number of Epochs ", Epochs)    
    start = time.time()
    print("Training \n")

    # Train Model
    for e in range(Epochs):
        training_loss = 0
        Model.train() # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(Train_loader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            Optimizer.zero_grad()
            
            # Forward and backward passes
            
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
        
            training_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(Model, valid_loader, Criterion ,Device)
            
                print("Epoch: {}/{} | ".format(e+1, Epochs),
                     "Training Loss: {:.4f} | ".format(training_loss/Print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(Test_loader)),
                     "Accuracy: {:.4f}".format(accuracy/len(Test_loader)))
            
                training_loss = 0
                Model.train()
                
    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    return Model

def test_model(Model, Test_loader, Device):
   # on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Test_loader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy  on test images is: %d%%' % (100 * correct / total))

def save_checkpoint(Model, Save_Dir, Train_data,Dropout=0.5,lr=0.001,Epoch=10):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'fc1':102,
                          'learning_rate':lr,
                          'dropout':Dropout,
                          'nb_of_epochs':Epoch,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')

        else: 
            print("Directory not found")
            
            

# =============================================================================
# Main Function
# =============================================================================         
def main():
     
    # Keyword Args for Training
    args = arg_parser()
    
    #Directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transformers 
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    train_loader = data_loader(train_data)
    valid_loader = data_loader(valid_data, train=False)
    test_loader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(structures=args.structures)
    
    # Build Classifier
    model.classifier = initial_classifier(model,fc1=args.fc1)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        #print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 50
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = train_model(model, train_loader, valid_loader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps,valid_loader)
    
    print("\nTraining process is now complete!!")
    #time_elapsed = time.time() - start
    #print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    
    # Let us Validation
    test_model(trained_model, test_loader, device)
    
    # Save the model
    save_checkpoint(trained_model, args.save_dir, train_data)



# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
    
    

