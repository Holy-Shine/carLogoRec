import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import numpy as np
import pickle as pkl
import h5py


dataset = h5py.File('../carDatasets.h5', 'r')

class LeNet5(nn.Module):
    def __init__(self,n_classes=5):
        super(LeNet5, self).__init__()
        # C1, S2   (1,32,32)-->(6,28,28)-->(6,14,14)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(4,4),stride=1,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # C3, S4   (6,14,14)-->(16,10,10)-->(16,5,5)
        self.conv3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=4,stride=1,padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # C5, F6   (16,5,5)-->()  (N,Cin,Hin,Win) ,
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,n_classes)
        
    def forward(self,input):
        feature_map1 = self.conv1(input)
        pooled_map1 = self.pool2(feature_map1)
        
        feature_map2 = self.conv3(pooled_map1)
        pooled_map2 = self.pool4(feature_map2)
        
        fc1 = F.relu(self.fc1(pooled_map2.view(-1,16*5*5)))
        fc2 = F.relu(self.fc2(fc1))
        out = F.log_softmax(self.fc3(fc2))
        return out
        
        
class carDataset(Dataset):
    def __init__(self, label='train'):
        self.dataset_X = np.array(dataset[label+'_X'])
        self.dataset_Y = np.array(dataset[label+'_Y'])
    def __len__(self):
        return self.dataset_X.shape[0]
    def __getitem__(self,idx):
        X = torch.FloatTensor(self.dataset_X[idx].reshape(1,32,32))
        Y = torch.LongTensor(self.dataset_Y[idx].reshape(-1))  # 改一下标签格式
        sample={'X':X,'Y':Y}
        return sample

def evaluate(model,test_set):

    test_X = test_set.dataset_X
    test_Y = test_set.dataset_Y

    with torch.no_grad():
        out = model(torch.FloatTensor(test_X.reshape(-1,1,32,32)))
        out = out.view(out.shape[0],-1) # 500 x 5
        prediction = torch.argmax(out, dim=1).unsqueeze(-1).view(1,-1) # 1x500
        real = torch.LongTensor(test_Y.reshape(1,-1))
        acc = 1.0*torch.sum(prediction==real).item()/5
        return acc

def train(model,num_epoch=10,batch_size=16,learning_rate=0.01,save=False):

    cardataset = carDataset(label='train')
    test_set = carDataset(label='test')
    dataloader = DataLoader(dataset=cardataset,batch_size=batch_size,shuffle=True,num_workers=0)

    optimizer = optim.Adam(params=model.parameters(),lr=learning_rate)
    loss_function = nn.NLLLoss()

    for epoch in range(num_epoch):
        for i_batch, batched_sample in enumerate(dataloader):
            X = batched_sample['X']
            Y = batched_sample['Y']
            optimizer.zero_grad()

            out = model(X)
            loss = loss_function(out.view(out.shape[0],-1),Y.view(-1))
            if i_batch % 10 == 0:
                acc=evaluate(model, test_set)
                print('loss after 10 batches:{}\tacc={}%'.format(loss,acc))
            loss.backward()
            optimizer.step()

    if save==True:
        saveModel(model)

# 存取模型
def saveModel(model):
    torch.save(model.state_dict(), '../ModelParams/cnn_lenet5.pkl')

def loadModel(model):
    model.load_state_dict(torch.load('../ModelParams/cnn_lenet5.pkl'))
    return model

def predict(model, img_path=None):
    labels=['雪铁龙','大众','一汽','福田','本田']
    from PIL import Image
    img_obj = Image.open(img_path).resize((32, 32)).convert('L')
    img_array = np.asarray(img_obj).reshape((1,1, 32 , 32))
    with torch.no_grad():
        X=torch.FloatTensor(img_array)
        out = model(X)
        label = torch.argmax(out,dim=1).item()
        print('这个车标是：%s'%labels[label])

model =LeNet5()
#train(model,save=True)
model=loadModel(model)
predict(model,img_path='../imgSample/Honda_sample.jpg')


