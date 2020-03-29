import argparse
import torch 
import helper as helper
import json
import numpy as np

def classify_image(path_to_image, checkpoint, top_k, category_names, gpu):
    if not torch.cuda.is_available() and gpu:
        raise("No gpu available to train the network. Please remove the --gpu argument to train using the cpu")
    device = ('cuda' if gpu else 'cpu')
        
    model = helper.load_model(checkpoint)
    image_tensor = torch.tensor(helper.load_image(path_to_image))
        
    (probs, classes) = helper.predict(image_tensor, model, top_k, device)
    if category_names != None:
        #convert the classes array to hold the string representation of the category
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[class_] for  class_ in classes]   

    return (classes, probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This scripts uses a chosen CNN-network to classify a given image')
    parser.add_argument('path_to_image', nargs = 1,help = 'Filelocation of the image to classify')
    parser.add_argument('checkpoint', nargs = 1,help = 'Checkpoint file of the CNN-network')
    parser.add_argument('--top_k', help = 'return the K most likely classes (default=2)', default = 2, type=int)
    parser.add_argument('--category_names', help = 'map the class integers to labels using the provided json dictonary', default = None)
    parser.add_argument('--gpu', help = "Use the gpu to accelerate inference", action = 'store_true')

    args = parser.parse_args()
    (classes, probs) = classify_image(args.path_to_image[0], args.checkpoint[0], args.top_k, args.category_names, args.gpu)

    for i in range(args.top_k):
        print("{0}: category : '{1}' with a probability of {2:.2f}%".format(i, classes[i], probs[i]*100))