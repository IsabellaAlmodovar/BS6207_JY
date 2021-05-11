import numpy as np
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transform

import dataset

import random
from sklearn import metrics
import matplotlib.pylab as plt
%matplotlib inline


def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def mydate() :
    return (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

datadir = 'training_data'
savemodeldir = 'new_model'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
savemodel = True
savemodel_interval = 1  #if 0 (and savemodel=True) will only save model at the end of entire training
loadmodel = False

#batch_size must be the multiple of 5
batch_size = 20
num_epochs = 20
lr = 0.001
log_interval = 1
random.seed(1) 


# preprocess
adataset = dataset.ProLigDataset(datadir,pos_ratio=300, neg_ratio=300)
bins = 48
hrange = 24
validationdataset = dataset.miniDataset(adataset.test_set, format='int',
                                        transform=transform.Compose([dataset.create_voxel2(bins=bins,hrange=hrange),
                                                                     dataset.array2tensor(torch.FloatTensor)]))
                                  
test_loader = torch.utils.data.DataLoader(dataset=validationdataset,
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)

print('length of dataset.positive:',len(adataset.positive))
print('length of dataset.nagetive:',len(adataset.negative))
print('length of valid set:',len(adataset.test_set))
print('length of training negative set:',len(adataset.negative_train))
print('length of training positive set:',len(adataset.positive_train))


def train(model,epoch):
    model.train() #put in training mode
    
    training_set = dataset.create_minidataset(adataset.negative_train, adataset.positive_train, 
                                              len(adataset.positive_train), epoch)

    minidataset = dataset.miniDataset(training_set,
                                      transform=transform.Compose([dataset.rotation3D(), #randomly rotate to augment 
                                                                   dataset.create_voxel2(bins=bins,hrange=hrange),
                                                                   dataset.array2tensor(torch.FloatTensor)]),
                                      target_transform=dataset.array2tensor(torch.FloatTensor))


    train_loader = torch.utils.data.DataLoader(dataset=minidataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=False)
    
    correct = 0
    total = 0
    train_loss = []
    for step, (inp,target) in enumerate(train_loader):
        target = target.float()
        inp, target = inp.to(device), target.to(device)
        outputs = model(inp)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        total += target.size(0)
        print(step, loss)
        correct += (outputs.ge(0.5).reshape(20, 1) == target.reshape(20, 1)).sum().item()
        acc = correct / total
    list_of_acc.append(acc)
    print('Accuracy of the model on the validation set: {} %'.format(100 * acc))
                
    print ('{:%Y-%m-%d %H:%M:%S} Epoch [{}/{}], Step [{}/{}] Loss: {:.6f}'.format( 
        datetime.now(), epoch+1, num_epochs, step+1, len(minidataset)//batch_size, loss.item()))
    
    list_of_losses.append(loss.item())
    
    #if (epoch+1) % log_interval == 0:
        #evaluate(model)
        #evaluate_mse(model)       
    if savemodel_interval != 0 and savemodel:
        if (epoch+1) % savemodel_interval == 0:
            torch.save(model.state_dict(),
                       '{}/{:%Y-%m-%d_%H-%M-%S}_model_epoch{}_step{}.pth'.format(savemodeldir,datetime.now(),epoch+1,step+1))
            print('model saved at epoch{} step{}'.format(epoch+1,step+1))


class CNN(nn.Module):
    # input size - the number of "classes"
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=0),
            #nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #print("in",x.shape)
        out = self.layer1(x)
        #print("layer 1 shape:",out.shape) 
        #print(out.shape)
        out = self.layer2(out)
        #print("layer 2 shape:",out.shape) 
        
        out = self.layer4(out)
        #print("layer 3 shape:",out.shape) 
        out = self.layer3(out)
        #print("layer 4 shape:",out.shape) 
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print("layer 5 shape:",out.shape) 
        #print(out.shape)
        out = self.fc1(out)
        #print("fc 1 shape:",out.shape) 
        #print(out.shape)
        out = self.fc2(out)
        #print("fc2 1 shape:",out.shape) 
        out = self.sigmoid(out)
        #print("out 1 shape:",out.shape) 
        #print(out.type())
        return out

       
cnn = CNN().to(device)
if loadmodel: # load checkpoint if needed
    print("Loading existing checkpoint...")
    cnn.load_state_dict(torch.load(loadmodelpath))
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
#loss
criterion = nn.MSELoss() 

list_of_losses = []
list_of_acc = []
for epoch in range(num_epochs):
    train(cnn,epoch)

if savemodel_interval == 0 and savemodel:
    torch.save(cnn.state_dict(), 
       '{}/{:%Y-%m-%d_%H-%M-%S}_model_epoch{}.pth'.format(savemodeldir,datetime.now(),num_epochs))
    print('model saved at epoch{}'.format(num_epochs))

name_1=['loss']
losses=pd.DataFrame(columns=name_1,data=list_of_losses)
print('losses',losses)
losses.to_csv('losses.csv',encoding='gbk')

name_2=['loss']
acc=pd.DataFrame(columns=name_2,data=list_of_acc)
print('acc',acc)
losses.to_csv('acc.csv',encoding='gbk')


plt.figure()
plt.plot(list_of_losses[:-1])
plt.savefig("loss.png")
plt.plot(list_of_acc[:-1])
plt.savefig("accuracy.png")

