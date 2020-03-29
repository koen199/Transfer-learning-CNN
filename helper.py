import torch 
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from PIL import Image
import numpy as np


def get_dataloaders(train_dir, valid_dir): 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
                                          ])


    validation_transforms = transforms.Compose([transforms.Resize(255),  
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
                                               ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    return (trainloader, validationloader, train_data.class_to_idx)




def get_model(architectue, learning_rate, hidden_units):
    model = None
    input_size = None 
    classifier_layer = None
    if architectue == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
        classifier_layer = 'classifier'
        
    elif architectue == "resnet18":
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
        classifier_layer = 'fc'
    else:
        raise("Unsupported network architecture. Supported architectures are 'vgg16' and 'Resnet18'")

                
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
                    nn.Linear(input_size, hidden_units, bias=True),
                    nn.ReLU(),                                
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units,2,bias=True),
                    nn.LogSoftmax(dim=1)
                )

    for param in classifier.parameters():
        param.requires_grad = True
        
    setattr(model, classifier_layer, classifier)  
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)    

    return (model, optimizer)





def validate(model, dataloader, criterion ,device):
    loss = 0
    accuracy = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        log_ps = model(images)
        loss += criterion(log_ps, labels).item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    loss = loss/len(dataloader)
    accuracy = accuracy/len(dataloader)
    return (loss, accuracy)
    


def train_model(model, trainloader, validationloader, optimizer, device, epochs):
    train_losses, validation_losses = [], []
    model.to(device)
    criterion = nn.NLLLoss()
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()
        model.eval()
        with torch.no_grad():
            (validate_loss, accuracy) = validate(model, validationloader, criterion, device)
            train_losses.append(running_loss/len(trainloader))
            validation_losses.append(validate_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(validate_loss),
              "Test Accuracy: {:.3f}".format(accuracy))  
        model.train()


def load_model(checkpoint_file):
    model = torch.load(checkpoint_file)
    model.eval()
    return model

def load_image(path_to_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(path_to_image)
    aspect_ratio = image.size[0]/image.size[1]
    #print(aspect_ratio)
    if aspect_ratio < 1:
        image.thumbnail((256/aspect_ratio, 256/aspect_ratio))
    else:
        image.thumbnail((256*aspect_ratio, 256*aspect_ratio))
    
    left =  (image.size[0]-224)/2
    right = image.size[0] - left
    upper = (image.size[1]-224)/2
    lower = image.size[1] - upper
    image = image.crop((left,upper,right,lower))
    
    np_image = np.array(image, dtype = float)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image/255
    np_image = (np_image - mean)/std    
    np_image = np_image.transpose((2,0,1))    
    return np_image


def predict(image_tensor, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        model.eval()
        image_tensor = torch.unsqueeze(image_tensor,0).float()
        image_tensor = image_tensor.to(device)
        model = model.to(device)
        log_ps = model.forward(image_tensor)
        probs, class_indices = torch.exp(log_ps).topk(topk, dim=1)
        idx_to_class = {model.class_to_idx[key]: key for key in model.class_to_idx}
        classes = [idx_to_class[class_index] for class_index in class_indices.cpu().numpy()[0]]
        probs = probs.cpu().numpy()[0]
        return probs , classes