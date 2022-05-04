import os
import torch

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.transforms import ToTensor

from PIL import Image

from cnn.predict import predict_image
from cnn.cnn_model import CnnModel
from utils.device import DeviceDataLoader, get_default_device, to_device


def train(): 
    data_path = os.path.join(os.getcwd(), 'dataset')
    dataset = ImageFolder(data_path + "/Training_test", transform=ToTensor())

    random_seed = 42

    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT


    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = int(len(dataset) * VAL_SPLIT)

    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.manual_seed(random_seed))
    
    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

    model = CnnModel()

    for images, labels in train_dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]', out[0])
        break

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)


    evaluate(model, val_dl)

    num_epochs = 3
    opt_func = torch.optim.Adam
    lr = 0.001

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    print(history)

    torch.save(model.state_dict, os.path.join(os.getcwd(), 'output/model.pth'))
    return ''

@torch.no_grad() 
def evaluate(model: CnnModel, val_loader: DataLoader):
    model.eval() # Setting model to evaluation mode, the model can adjust its behavior regarding some operations, like Dropout.
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
  
def fit(epochs, lr: float, model: CnnModel, train_loader: DataLoader, val_loader: DataLoader, opt_func=torch.optim.SGD):
    history=[]
    optimizer= opt_func(model.parameters(), lr) # model paramters w.r.t calculate derivative of loss
    for epoch in range(epochs):
        # Training phase
        model.train() # Setting model to training mode
        train_losses=[]
        for batch in train_loader:
            loss= model.training_step(batch)
            train_losses.append(loss)
            loss.backward() #compute  gradients
            optimizer.step()
            optimizer.zero_grad() # zero the gradients
        #Validation phase
        result= evaluate(model,val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def classificate_image(filename: str):
    data_path = os.path.join(os.getcwd(), 'dataset', 'Test')
    dataset = ImageFolder(data_path, transform=ToTensor())

    file_path = os.path.join(os.getcwd(), 'uploads', filename)
    image = Image.open(file_path)

    transform = ToTensor()
    tensor = transform(image)
  
    model = torch.load(os.path.join(os.getcwd(), 'output/model.pth'))
    device = get_default_device()

    # img,label= dataset[8]
    pred_img = predict_image(dataset, tensor, model, device)
    return pred_img