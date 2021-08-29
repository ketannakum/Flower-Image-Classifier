# PROGRAMMER: KETAN N.
# DATE CREATED: JULY 15th, 2021                           
# REVISED DATE: JULY 23rd, 2021
# PURPOSE: This program purpose is to use a trained network for inference. 
#          That is, you'll pass an image into the network and predict the 
#          class of the flower in the image. 
#
#          The program is broken down into multiple steps:
#          1. Load the custom pretrained model from checkpoint
#          2. Preprocess image so that it can be used in prediction
#          3. Predict image
#          
#          Basic usage: python predict.py /path/to/image checkpoint
#          Options:
#               Return top K most likely classes: python predict.py input checkpoint --top_k 3
#               Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#               Use GPU for inference: python predict.py input checkpoint --gpu

# Imports pythom modules
from time import time, sleep
import argparse
#import helper
import numpy as np
import seaborn as sb
from PIL import Image
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
#import torch.nn.functional as F
#from torch.autograd import Variable
#from collections import OrderedDict

# A function to predict the class from an image file path
def predict(image_path, model, cat_to_name, gpu, topk):
    """ 
    Predict the class (or classes) of an image using a trained deep learning model.
    Parameters: 
        image_path - The image path and image which should be used for predicting
        model - Trained deep learning model which will be used for prediction
        cat_to_name - This will have a dictionary mapping the integer encoded
                      categories to the actual names of the flowers. (Label Mapping)
        topk - top K classes probability for a given image 

    Returns:
        None - This function will predict topk classes for given image and print:   
               Image Classes, 
               Image Classes Prediction and 
               Flower Names (optional - based on inputs to the program)
    """
    
    # Process image
    img = process_image(image_path)
    
    # Convert Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    #print(image_tensor)
    
    # Add batch dimension since pytorch treats all images as batches
    model_input = image_tensor.unsqueeze(0)
    #print("Image shape for prediction :", model_input.shape)
    #model_input = img.view(1, img)
        
    if gpu:
        if torch.cuda.is_available():
            model_input.cuda()
        else:
            model_input.cpu()
    else:
        model_input.cpu()

    # Probability
    ps = torch.exp(model.forward(model_input))
    
    # Top probs (returns as tensor)
    top_k_values, top_k_indices = ps.topk(topk,dim=1)
    
    # Convert to list
    top_k_values = top_k_values.detach().numpy().tolist()[0] 
    top_k_indices = top_k_indices.detach().numpy().tolist()[0]
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_k_classes = [idx_to_class[key] for key in top_k_indices]
    
    print(" ")
    print("Top {} Image Class or Classes {} ".format(topk, top_k_classes))
    print("Top {} Class or Classes Prediction {} ".format(topk, top_k_values))
    
    if cat_to_name != None:
        # Convert indices to classes
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        #print(cat_to_name)
        
        #top_flowers_labels = [cat_to_name[idx_to_class[labels]] for labels in top_k_indices]
        top_flowers_labels = [cat_to_name[key] for key in top_k_classes]
        print("Top {} Flower Names {} ".format(topk, top_flowers_labels))


# A function that process a PIL image for use in a PyTorch model
def process_image(image_path):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    Parameters: 
        image_path - An image which should be processed to use in prediction
              
    Returns:
        np_image - Return an Numpy array which can be used for prediction
    """
    
    # Open image using image path
    image = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = image.size
    
    #print("Original Size: ", image.size)
    
    # Scale/ Resize the image
    if width > height:
        image.thumbnail((25000, 256))
    else:
        image.thumbnail((256, 25000))
    
    # Updated the dimensions of the image
    width, height = image.size
    
    #print("After Scale/ Resize and Before Crop: ", image.size)
    
    # Do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    #print("After Crop: ", image.size)
    
    # Turn image into numpy array
    np_image = np.array(image)
    
    # Make all values between 0 and 1
    np_image = np_image/255
    
    # Normalize based on the mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    # Make the color channel dimension first instead of last
    np_image = np_image.transpose((2, 0, 1))
    
    # Print image shape and it should look as (3, 244, 244)
    #print("Image shape after preprocessing: ", np_image.shape)
    
    return np_image

# A function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, gpu):
    """ 
    A function that loads a checkpoint and rebuilds the model
    Parameters: 
        filepath - An filepath including filename pf checkpoint which should be loaded
        gpu - True means gpu will be used to test images 
              False means cpu will be used to test images      
    Returns:
        model - Return pretrained model for prediction
    """
    
    # Use GPU if it's available
    if gpu:
        print("*** Loading checkpoint in cuda *** \n ")
        checkpoint = torch.load(filepath)
    else: 
        print("*** Loading checkpoint in cpu *** \n ")
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    print("{:45}: {}".format('Loading model with arch', checkpoint['arch']))
    print(" ")
    
    # Load pretrained model according to arch of the checkpoint
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        print("\n Sorry base architecture not recognized \n")
        exit()
    
    # Create the classifier 
    if checkpoint['arch'] in ['vgg13', 'vgg16', 'vgg19']:
        layers = []
        layers.append(nn.Linear(checkpoint['input_size'], 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        if checkpoint['hidden_layer_input'] != 0:
            layers.append(nn.Linear(4096, checkpoint['hidden_layer_input']))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(checkpoint['hidden_layer_input'], checkpoint['output_size']))
        else:
            layers.append(nn.Linear(4096, checkpoint['output_size']))
        layers.append(nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(*layers) 
    elif checkpoint['arch'] in ['resnet34']:
        layers = []
        layers.append(nn.Linear(checkpoint['input_size'], 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(256, checkpoint['output_size']))
        layers.append(nn.LogSoftmax(dim=1))
        model.fc = nn.Sequential(*layers)
        if checkpoint['hidden_layer_input'] != 0:
            print(" *** No custom hidden units with this pretrained model! ***")
    elif checkpoint['arch'] in ['alexnet']:
        layers = []
        layers.append(nn.Linear(checkpoint['input_size'], 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        if checkpoint['hidden_layer_input'] != 0:
            layers.append(nn.Linear(4096, checkpoint['hidden_layer_input']))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(checkpoint['hidden_layer_input'], checkpoint['output_size']))
        else:
            layers.append(nn.Linear(4096, checkpoint['output_size']))
        layers.append(nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(*layers)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
     
    epochs = checkpoint['epochs']
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = checkpoint['optimizer']
    optimizer.state_dict = checkpoint['optimizer_state_dict']
    
    #print("epochs: \n\n", epochs, '\n')
    #print("optimizer: \n\n", optimizer.state_dict, '\n')
    
    return model

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments where applicable. 
    Command Line Arguments:
      1. Image file with path as input for prediction
      2. Model checkpoint path with filename as checkpoint 
      3. Number of predicition for input image as --top_k
      4. File which has dictionary for flower names as --category_names
      5. Gpu as --gpu to be used for training model. If present in command line then gpu will be used
    This function returns these arguments as an ArgumentParser object.
    Parameters:
        None - simply using argparse module to create & store command line arguments
    Returns:
        parse_args() -data structure that stores the command line arguments object  
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Argument 1: arg for image file (with path) as the input for prediction
    parser.add_argument('input', action = "store", help = 'Path to Image including image name') 
    
    # Argument 2: arg fo load model from checkpoint for prediction
    parser.add_argument('checkpoint', action = "store", help = 'Model checkpoint path including filename') 
    
    # Argument 3: arg for top_k probablities for prediction
    parser.add_argument('--top_k', type = int, default = 1, help = 'Number of predictions for an image. By default it will return top 1                                 prediction for an image')
    
    # Argument 4: arg for filename which contains dictionary of index and flower name
    parser.add_argument('--category_names', type = str, default = None, help = 'File name which has dictionary for flower names')
    
    # Argument 5: arg for gpu to be used for training model
    parser.add_argument('--gpu', action='store_const', const=True, default=False, help = 'Enable GPU for training model') 
    
    # return args which are parsed to calling module
    return parser.parse_args()

# Main program function defined below
def main():
    
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg 
    #print(sys.argv[1])
    in_arg = get_input_args()
    
    # Print input arguments which will be used for loading model and prediction of an image
    print("\n *** Input arguments for this execution *** \n", in_arg)
    print(" ")
    #print(in_arg)
    
    #device = torch.device('cpu')
    
    # Load model from a checkpoint
    model = load_checkpoint(in_arg.checkpoint, in_arg.gpu)

    print("\n *** Custom model base architecture *** \n", model)
    #print(model)
    
    # Model evaluation mode
    model.eval()
    
    # Predict image using model
    predict(in_arg.input, model, in_arg.category_names, in_arg.gpu, in_arg.top_k)
    
    #image_path = in_arg.input
    #probs, classes, labels = predict(in_arg.input, model, cat_to_name, in_arg.gpu, in_arg.top_k)
    #print("Probability :", probs)
    #print("Classes : ", classes)
    #print("Labels : ", labels)
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n*** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(round((tot_time%3600)%60,2)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()