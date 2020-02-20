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
    
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created in train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    
    #  top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help=' top k classes.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='names of structures/architectures.')

    # Add GPU 
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='GPU ')

    # Parse args
    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load("my_checkpoint.pth")
    
    # Load Defaults if none specified
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    data_dir = "./flowers"
    image = (data_dir + '/test' + '/1/' + 'image_06752.jpg')
    pil_image = PIL.Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(pil_image)
    np_image = np.array(img_tensor)
    
    return np_image


def predict(image_tensor, model, device, cat_to_name, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    
    # check top_k
    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)

    model=model.cpu()

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

 
# =============================================================================
# Main Function
# =============================================================================       
def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # The top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)
    
    # Print out probabilities
    print_probability(top_flowers, top_probs)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()







