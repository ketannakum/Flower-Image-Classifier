# PROGRAMMER: KETAN N.
# DATE CREATED: JULY 9th, 2021                           
# REVISED DATE: JULY 23rd, 2021
# PURPOSE: This program purpose is to load a pretrained model 
#          and customize the classifier parameters so that it can 
#          be trained to recognize different species of flowers.
#
#          The program is broken down into multiple steps:
#          1. Load and preprocess the image dataset
#          2. Train the image classifier on your dataset
#          3. Validate the model with test dataset
#          4. Save the model as a checkpoint
#          
#          Basic usage: python train.py data_directory
#          Prints out training loss, validation loss, and validation accuracy as the network trains
#          Options:
#               Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#               Choose architecture: python train.py data_dir --arch "vgg13"
#               Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#               Use GPU for training: python train.py data_dir --gpu

# Imports python modules
from time import time, sleep
import argparse
import sys
import os
from workspace_utils import active_session
import torch
from torch import optim
from torchvision import datasets, transforms, models
from torch import nn
#import torch.nn.functional as F
#from torch.autograd import Variable
#from collections import OrderedDict


# Run the test images through the network and measure the accuracy, the same way validation phase is run.
def test_validation(model, dataloaders, criterion, gpu):
    """
     Run the test images through the network and measure the accuracy, the same way validation phase is run.
     Parameters: 
        model - The trained model for test data validation
        dataloaders - Data set of with images and labels as tensor
        criterion - Compute a gradient according to a given loss function
        gpu - True means gpu will be used to process test images 
              False means cpu will be used to process test images
     Returns:
        None - This function will just validate the test images using the model 
               which was trained.
    """
    
    #if gpu:
    #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use GPU if it's available
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    print("{:45}: {}".format('Device using cuda or cpu', device))
    print(" ")
    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")
            model.train()

# Train model through the network, run validation images through network and measure the accuracy.
def train(model, dataloaders, image_datasets, learning_rate, gpu, arch, hidden_layer_input, ckpt_path, num_of_epochs, print_every_input=10):
    """
     Train model through the network, run validation images through network and measure the accuracy.
     Parameters: 
        model - Train model by using train and validation datasets 
        dataloaders - Data set with images and labels as tensor
        image_datasets - Image data set after applying transforms 
        gpu - True means gpu will be used to test images 
              False means cpu will be used to test images
        arch - Architecture which is being used to train model
        hiddent_layer_input - Hidden layer inputs received from command line argement which will be used in saving checkpoint
        chpt_path - Checkpoint path received from command line argument which will be used in saving checkpoint
        num_of_epochs - Epochs to be run for training a model
        print_every_input - Argument to control after how many steps the validation will be performed and results will be printed
     Returns:
        model - Trained model
    """
    
    # Measures training model runtime by collecting start time
    start_time = time()
    
    #if gpu:
    #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use GPU if it's available
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    if arch in ['vgg13', 'vgg16', 'vgg19', 'alexnet']:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch in ['resnet34']:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    model.to(device)
    
    print("\n\n*** Starting Training *** \n")
    print("{:45}: {}".format('Device using cuda or cpu', device))
    print("{:45}: {}".format('Learning Rate', learning_rate))
    print("{:45}: {}".format('Checkpoint Save Path', ckpt_path))
    print("{:45}: {}".format('Number of epochs', num_of_epochs))
    print("{:45}: {}".format('Training model with arch', arch))
    if arch in ['vgg13', 'vgg16', 'vgg19', 'alexnet']:
        print("{:45}: {}".format('Hidden layer inputs', hidden_layer_input))    
    elif arch == 'resnet34':
        print("{:45}: {}".format('Hidden layer inputs', 0))
    print(" ")
    
    # Train model
    with active_session():
        # do long-running work here
        epochs = num_of_epochs
        steps = 0
        running_loss = 0
        print_every = print_every_input

        for epoch in range(epochs):
            for inputs, labels in dataloaders['train']:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in dataloaders['valid']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                          f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                    running_loss = 0
                    model.train()
                    
        # Measure training model runtime by collecting end time
        end_time = time()            
        
        # Computes overall runtime in seconds & prints it in hh:mm:ss format
        tot_time = end_time - start_time
        print("\n *** Total Elapsed Runtime for training model:",
              str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
              +str(round((tot_time%3600)%60,2)) )
        
        print("\n*** Ending Training ***\n")
        
        print("*** Test Data Validation *** \n")
        test_validation(model, dataloaders, criterion, gpu)
        
        # Preparation for saving the checkpoint 
        model.class_to_idx = image_datasets['train'].class_to_idx
        #print("model class to idx : \n", model.class_to_idx) 
        
        if arch in ['vgg13', 'vgg16', 'vgg19']:
            checkpoint = {'arch': arch,
                          'epochs': num_of_epochs,
                          'input_size': 25088,
                          'hidden_layer_input': hidden_layer_input,
                          'output_size': 102,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer,
                          'optimizer_state_dict': optimizer.state_dict,
                          'class_to_idx': model.class_to_idx}
        elif arch in ['resnet34']:
            checkpoint = {'arch': arch,
                          'epochs': num_of_epochs,
                          'input_size': 512,
                          'hidden_layer_input': hidden_layer_input,
                          'output_size': 102,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer,
                          'optimizer_state_dict': optimizer.state_dict,
                          'class_to_idx': model.class_to_idx}
        elif arch in ['alexnet']:
            checkpoint = {'arch': arch,
                          'epochs': num_of_epochs,
                          'input_size': 9216,
                          'hidden_layer_input': hidden_layer_input,
                          'output_size': 102,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer,
                          'optimizer_state_dict': optimizer.state_dict,
                          'class_to_idx': model.class_to_idx}
            
        name = "checkpoint"
        path = os.path.join(ckpt_path, 'nn_{}_{}_{}.pth'.format(arch, name, num_of_epochs))
        
        #print("\n Saving checkpoint at {:25}".format(path))
        print(" ")
        print("{:45}: {}".format('Saving checkpoint at', path))
        #print("Save checkpoint path and name", path)
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
    return model

# Load pretrained model which can be trained for flowers data set
def load_model(arch, hidden_layer_input):
    """
     Load pretrained model which can be trained for flowers data set.
     Parameters: 
        arch - Architecture which is being used to load the pretrained model
        hiddent_layer_input - Hidden layer inputs received from command line argument which will be used in training model
     Returns:
        model - Pretrained model
    """

    # Download pretrained model according to the command line agrument input for arch
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        print("\n Sorry base architecture not recognized")
        exit()
    
    print("\n *** Pretrained model base architecture *** \n", model)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Customize the classifier according to flowers data set goals
    if arch in ['vgg13', 'vgg16', 'vgg19']:
        layers = []
        layers.append(nn.Linear(25088, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        if hidden_layer_input != 0:
            layers.append(nn.Linear(4096, hidden_layer_input))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(hidden_layer_input, 102))
        else:
            layers.append(nn.Linear(4096, 102))
        layers.append(nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(*layers)
    elif arch in ['resnet34']:
        layers = []
        layers.append(nn.Linear(512, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(256, 102))
        layers.append(nn.LogSoftmax(dim=1))
        model.fc = nn.Sequential(*layers)
        if hidden_layer_input != 0:
            print("\n *** Sorry can't use hidden units with this pretrained model! ***")
    elif arch in ['alexnet']:
        layers = []
        layers.append(nn.Linear(9216, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        if hidden_layer_input != 0:
            layers.append(nn.Linear(4096, hidden_layer_input))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(int(hidden_layer_input), 102))
        else:
            layers.append(nn.Linear(4096, 102))
        layers.append(nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(*layers)
    
    print("\n *** Custom model base architecture *** \n", model)
    
    return model
             
# Load data set for training model
def get_data(image_dir):
    """
     Load data set for training model.
     Parameters: 
        image_dir - Image directory from where data sets will be loaded
     Returns:
           None - This function will just validate the test images using the model 
                    which was trained.
    """
    
    data_dir = image_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder
    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test' : test_dir}
    
    # Load image data set using ImageFolder and transforms
    image_datasets = {x: datasets.ImageFolder(dirs[x], 
                     transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

    # Using the image datasets, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) 
                   for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) 
                                  for x in ['train', 'valid', 'test']}
    
    return dataloaders, dataset_sizes, image_datasets, dirs

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments where applicable. 
    Command Line Arguments:
      1. Image folder as data_dir with no default value
      2. Path to a folder as --save_dir where trained model will be saved as checkpoint
      3. Torchvision pretrained model as --arch to be used for training model
      4. Learning rate as --learning_rate to be used for training model
      5. Hiddent units as --hidden_units to be used for training model. If 0 then no hidden units will be used for training model
      6. Epochs as --epochs to be used for training model
      7. Gpu as --gpu to be used for training model. If present in command line then gpu will be used
    This function returns these arguments as an ArgumentParser object.
    Parameters:
        None - simply using argparse module to create & store command line arguments
    Returns:
        parse_args() -data structure that stores the command line arguments object  
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Argument 1: arg for a path to a folder from where data set will be loaded for training model
    parser.add_argument('data_dir', action = "store", help = 'Data Directory with flower images for training model') 
    
    # Argument 2: arg for a path to a folder where trained model will be saved
    parser.add_argument('--save_dir', type = str, default = os.getcwd(), help = 'Path to save checkpoint') 

    # Argument 3: arg for torchvision.model architecture to be used for training
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Torchvision pretrained model to be used for training. Supported models                         are vgg13, vgg16, vgg19, resnet34 and alexnet') 
    
    # Argument 4: arg for learning rate to be used for training model
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate for training model') 
    
    # Argument 5: arg for hidden units to be used for training model
    parser.add_argument('--hidden_units', type = int, default = 0, help = 'Hidden Layer units for training model. If 0 then no additional hidden                        layer will be used')
    
    # Argument 6: arg for epochs to be used for training model
    parser.add_argument('--epochs', type = int, default = 5, help = 'Epochs for training model')
    
    # Argument 7: arg for gpu to be used for training model
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
    
    # Print input arguments which will be used for loading, training and saving model
    print(" ")
    print("*** Input arguments for this execution *** \n", in_arg)
    #print(in_arg)
    
    # Set data variables using get_data function
    dataloaders, dataset_sizes, image_datasets, dirs = get_data(in_arg.data_dir)
    
    # Verify train images shape and type. Verfy label shape. 
    #dataiter = iter(dataloaders['train'])
    #images, labels = dataiter.next()
    #print(type(images))
    #print(images.shape)
    #print(labels.shape)
    
    # Verify dataset for train, valid and test
    print()
    for x in dataset_sizes:
        print("Loaded {} images under {}".format(dataset_sizes[x],x))
    
    # Load model for training
    model = load_model(in_arg.arch, in_arg.hidden_units)
    
    # Train model
    model = train(model, dataloaders, image_datasets, in_arg.learning_rate, in_arg.gpu, in_arg.arch, 
                  in_arg.hidden_units, in_arg.save_dir, in_arg.epochs)
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n *** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(round((tot_time%3600)%60,2)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()