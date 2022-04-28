# import torch
import os
import torch

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from cnn.predict import predict_image
from cnn.cnn_model import CnnModel
from utils.device import DeviceDataLoader, get_default_device, to_device


def train(): 
    data_path = os.path.join(os.getcwd(), 'dataset')
    data_dir_list = os.listdir(data_path)
    dataset = ImageFolder(data_path + "/Training", transform=ToTensor)
    print(dataset.classes)


    random_seed = 42

    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.manual_seed(random_seed))

    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_dl = DataLoader(val_ds.dataset, batch_size, num_workers=4, pin_memory=True)

    model = CnnModel()

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    # val_dl = DeviceDataLoader(val_dl, device)
    # to_device(model, device)

    for images, labels in train_dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:',out.shape)
        print('out[0]',out[0])
        break

    num_epochs = 15
    opt_func = torch.optim.Adam
    lr = 0.001

    # history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    test_dataset = ImageFolder(data_path + "/Test", transform=ToTensor())

    img,label= test_dataset[8]
    pred_img = predict_image(dataset, img, model, device)
    return pred_img

@torch.no_grad() 
def evaluate(model, val_loader):
    model.eval() # Setting model to evaluation mode, the model can adjust its behavior regarding some operations, like Dropout.
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
  
def fit(epochs, lr, model, train_loader, val_loader, opt_func= torch.optim.SGD):
    history=[]
    optimizer= opt_func(model.parameters(),lr) # model paramters w.r.t calculate derivative of loss
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