import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Définir le modèle SimpleCNN

def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)



# Neural Network
class CNN_Network(nn.Module):
    
    # Network Initialisation
    def __init__(self, params):
        
        super(CNN_Network, self).__init__()
    
        Cin,Hin,Win=params["shape_in"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]  
        num_classes=params["num_classes"] 
        self.dropout_rate=params["dropout_rate"] 
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = F.relu(self.fc1(X))
        X=F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)
    
# import torch.optim as optim
#     # Instancier le modèle
# params_model={
#     "shape_in": (1,128,128), 
#     "initial_filters": 8,    
#     "num_fc1": 100,
#     "dropout_rate": 0.25,
#     "num_classes": 1}
# model = CNN_Network(params_model)

# # Définir la fonction de perte et l'optimiseur
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Exemple de données d'entrée et de cibles
# input_data = torch.randn(16, 1, 128, 128)  # 16 exemples avec des images en noir et blanc de taille 128x128
# targets = torch.randint(0, 2, (16, 1)).float()  # 16 cibles (0 ou 1)

# print(targets.shape, input_data.shape)

# # Avant-propagation
# output = model(input_data)

# print(output)

# # Calcul de la perte
# loss = criterion(output, targets)

# # Rétropropagation
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(f"Loss: {loss.item()}")
    