import numpy as np
import math
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transform
import dataset
import pickle
import matplotlib.pylab as plt



def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def mydate() :
    return (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


datadir = 'testing_data_release/testing_data'
loadmodelpath = '/content/drive/MyDrive/BS6207ProjectJiaoYan/new_model/2021-05-11_11-08-10_model_epoch19_step270.pth'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loadmodel = True

batch_size = 1
log_interval = 5000
lr = 0.001
bins = 48
hrange = 24

# =dataloader
adataset = dataset.ProLigDataset(datadir,split=False)

testdataset = dataset.miniDataset(dataset=adataset.all_pairs,mode='test',
                                        transform=transform.Compose([dataset.create_voxel2(bins=bins,hrange=hrange),dataset.array2tensor(torch.FloatTensor)]))

test_loader = torch.utils.data.DataLoader(dataset=testdataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=4,
                                          drop_last=False)

print("size of dataset is",len(testdataset))
print("no. of steps per epoch is",len(adataset)//batch_size)

for i in range(len(testdataset)):
    sample,target = testdataset[i]
    #plt.plot(sample['audio'])
    print(sample.shape)
    #print(target.shape)
    if i == 1:
        break
        
#print(adataset.all_pairs[0+824:824+824])
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
# load trained model
cnn = CNN().to(device)
if loadmodel: # load checkpoint if needed
    print("Loading existing checkpoint...")
    cnn.load_state_dict(torch.load(loadmodelpath))

# Predict!
predictions = []
def evaluate_mse(model):
    model.eval()
    with torch.no_grad():
        for step, (inp, _) in enumerate(test_loader):
            inp = inp.to(device)
            outputs = model(inp)
            outputs_numpy = outputs.detach().cpu().numpy()
            for i in range(outputs_numpy.shape[0]):
                predictions.append(outputs_numpy.item(i))

            if (step+1) % log_interval == 0:
                save_obj(predictions,"pred")
                print ('{:%Y-%m-%d %H:%M:%S} Step [{}/{}]'.format(datetime.now(), step+1, len(testdataset)//batch_size))  
print('{:%Y-%m-%d %H:%M:%S} Starting prediction...'.format(datetime.now()))
start_time = time.monotonic()
evaluate_mse(cnn)
elapsed_time = time.monotonic() - start_time
print('Testing time taken:',time_taken(elapsed_time))

def out_format(predictions,dim,top):
    """predictions - list of output predictions
       dim - no of samples in dataset
       top - find top (number) of positive examples"""
    assert len(predictions) == dim**2
    pred_ = np.array(predictions)
    #print(pred_[:dim])
    b = np.array(pred_).reshape((dim,dim))
    idx = (np.abs(b - 1))
    #print(idx)
    for row in range(idx.shape[0]):
        q = np.argpartition(idx[row], top)
        #print(q)
        count = 0
        for index in q:
            idx[row][index] = 1
            count += 1
            if count == top:
                break

    idx[idx!=1]=0
    #print(idx)
    #print(idx.shape)
    idx = idx.reshape(dim**2)
    #print(idx.shape)
    return list(idx)

pred = out_format(predictions,824,10)
print('pred[:824]:',pred[:824])

import pandas as pd

print(len(adataset.all_pairs))
save_obj(predictions,"pred")
save_obj(adataset.all_pairs,"test_dataset")
pred = load_obj("pred")
pairs = np.load('test_dataset.pkl', allow_pickle=True)
print(len(pairs))
result = pred 
result = np.load('pred.pkl',  allow_pickle=True)
print(len(result))
base=1
step=824
output=[]
output_step=[]
counter = 0
protein_numbers = range(1, 825)
output_dict = {}

for x in range(0,len(result)):
    if result[x] == 1.:
        output_step.append(str(pairs[x][1][34:38]).lstrip('0'))  #append the ligand number
        counter += 1
    if counter == 10 or x == step*base:
        if x == step*base:
            base += 1
        else:
            x = step*base-1
            base += 1
        #print(pairs[x])
        counter = 0
        output_step += ['NA']*(10-len(output_step))
        #print(output_step)
        output.append(output_step)
        output_step = []

output_dict.update({'pro_id': protein_numbers})
counter = 1
output_transposed = np.transpose(output)
#print(output_transposed.shape)

for x in output_transposed:
    output_dict.update({'lig'+str(counter)+'_id': x})
    #print(len(x))
    counter += 1

#print(output_dict)
output_panda = pd.DataFrame(data=output_dict)
output_panda.to_csv('test_predictions.txt', sep='\t',index=False, header=True)        
#print(output_panda)
