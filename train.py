import torch 
import argparse
import helper as helper

parser = argparse.ArgumentParser(description='This scripts trains a CNN-network to classify images')
parser.add_argument("data_directory", nargs = 1,help = 'The directory where the training and validation data is stored')
parser.add_argument('--save_dir', help = 'The directory where the network will be stored after training. Not stored if not argument not passed.', default = None)
parser.add_argument('--arch', help = "The architecture type of the CNN-network. 'vgg16' and 'resnet18' are supported (default='vgg16')", default = 'vgg16')
parser.add_argument('--learning_rate', help = "The learning rate used by the optimizer to train the network (default=0.001)", default = 0.001)
parser.add_argument('--hidden_units', help = "Amount of hidden units in the hidden layer of the classifier part of the CNN-network (default=4096)", default = 4096, type=int)
parser.add_argument('--epochs', help = "Amount of epochs the network is trained (default=5)", default = 5,type=int)
parser.add_argument('--gpu', help = "Use the gpu to accelerate training", action = 'store_true')

args = parser.parse_args()
train_dir = args.data_directory[0] + '/train'
valid_dir = args.data_directory[0] + '/valid'

(trainloader, validationloader, class_to_idx) = helper.get_dataloaders(train_dir, valid_dir)

if not torch.cuda.is_available() and args.gpu:
    raise("No gpu available to train the network. Please remove the --gpu argument to train using the cpu")
device = ('cuda' if args.gpu else 'cpu')
    
    
(model, optimizer) = helper.get_model(args.arch, args.learning_rate, args.hidden_units)
helper.train_model(model, trainloader, validationloader, optimizer, device, args.epochs)


if args.save_dir != None:
    model.class_to_idx = class_to_idx
    torch.save(model, args.save_dir)

  