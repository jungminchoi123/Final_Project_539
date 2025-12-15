#%% Load parameters

import torch
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,matthews_corrcoef

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve, auc

from scipy import interpolate

import Utils
from Utils.imgtools import getimgdf,GetDatasetSplit
from Utils.getdataloader import GetDataLoader, GetTrainValidDataLoader

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time

#%%

print('Experiment using SGD as optmizer')


start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"{device} is available")
print(torch.cuda.get_device_name(device=None)) # GPU Available

#%%

# # Local setup
batch_size = 128
target_size = (224,224)
num_epochs = 100

traindatasetpath = 'train/' 
valdatasetpath = 'val/' 
testdatasetpath = 'test/' 


classes = ['normal','pneumonia','tuberculosis']
num_classes = len(classes) 

# Create a DataFrame of all the data
trainingimgdataframe = getimgdf(traindatasetpath,classes)
validationimgdataframe = getimgdf(valdatasetpath ,classes)
testingimgdataframe = getimgdf(testdatasetpath,classes)

trainingset_df,_ = GetDatasetSplit(trainingimgdataframe,
                                             0,classes)

validationset_df,_ = GetDatasetSplit(validationimgdataframe,
                                     0,classes)

testingset_df,_ = GetDatasetSplit(testingimgdataframe,
                                     0,classes)

# Transform 3 classes to 2 classes for binary classification

classes = ['normal','infection']

trainingset_df['label'][trainingset_df['label']==2]=1
validationset_df['label'][validationset_df['label']==2]=1
testingset_df['label'][testingset_df['label']==2]=1
# Shuffling Training set
trainingset_df = trainingset_df.sample(frac=1).reset_index(drop=True)
validationset_df = validationset_df.sample(frac=1).reset_index(drop=True)
testingset_df = testingset_df.sample(frac=1).reset_index(drop=True)

# trainingset_df = trainingset_df.iloc[:100]
# validationset_df = validationset_df.iloc[:50]
# testingset_df = testingset_df.iloc[:50]


#%% Create DataLoader Training, Validation and Test Data Loader

print('------------------------')
print('Data Partition')
print('------------------------')


traindataloader = GetDataLoader(trainingset_df[['imgpath','label']],
                            target_size,
                            classes,
                            batch_size,
                            data_augm=True,
                            setname='Training set')

validationdataloader = GetDataLoader(validationset_df[['imgpath','label']],
                            target_size,
                            classes,
                            batch_size,
                            data_augm=False,
                            setname='Validation set')

testingdataloader = GetDataLoader(testingset_df[['imgpath','label']],
                            target_size,
                            classes,
                            batch_size,
                            data_augm=False,
                            setname='Hold-out Testing')

print('------------------------')
print('------------------------')

#%% Show samples from train loader

showimagesample = True

if showimagesample:
    examples = next(iter(traindataloader))
    # examples = DataAugmentation(examples)
    labels = examples[1]
    examples = examples[0].permute((0,2,3,1))
    
    # plt.subplot(4,4,i)
    plt.figure()
    
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(examples[i,:,:,:],cmap='gray')
        
        if labels[i].item() == 0:
            plt.title('Normal')
            
        if labels[i].item() == 1:
            plt.title('Pneumonia')
        
        plt.axis('off')

#%%

from torchvision import models

#%%
import torch.nn as nn
import torch.optim as optim
# Define your model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
"""
class BinaryCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Load pre-trained ResNet-50
        self.model = models.resnet50(pretrained=None)
        
        # Get number of features in the final layer
        num_ftrs = self.model.fc.in_features
        
        # Replace the final fully connected layer
        # For binary classification with sigmoid, we use 1 output unit
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        
        # if x.shape[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)  # Repeat channel dimension
    
        return torch.sigmoid(x)  # Output: probability between 0-1
        
        # return torch.softmax(out1, dim=1)  # Use softmax for 2 classes
"""
class BinaryCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(BinaryCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # After 4 maxpool layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # 2 classes
        
    def forward(self, x):
        # Input: (batch_size, 1, 224, 224)
        #nn.PReLU()
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Output: (batch_size, 32, 112, 112)
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Output: (batch_size, 64, 56, 56)
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Output: (batch_size, 128, 28, 28)
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Output: (batch_size, 256, 14, 14)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # Output: (batch_size, 256 * 14 * 14) = (batch_size, 50176)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        # Output: (batch_size, 2)
        
        return torch.sigmoid(x)

#%% 

dummy_input = torch.randn(16, 3, 224, 224)

# Initialize model
model = BinaryCNN().to(device)

# Forward pass
output = model(dummy_input.to(device))

#%%

# Initialize model, loss, optimizer

criterion = nn.BCEWithLogitsLoss() 
# criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01)
#optimizer = optim.Adadelta(model.parameters(), lr=1.0)

#%%

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

model.train()
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

print("---- Model Training and Validation---- \n")
print("-" * 15)


for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for data, targets in traindataloader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        outputs = outputs.squeeze(1)
        
        # Ensure both are float
        outputs = outputs.float()
        targets = targets.float()
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = ((outputs>0.5).int()).float()
        train_correct += (predicted == targets).sum().item()
        train_total += targets.size(0)
        
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, targets in validationdataloader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            outputs = outputs.squeeze(1)
            
            outputs = outputs.float()
            targets = targets.float()
            
            loss = criterion(outputs, targets.float())
            
            val_loss += loss.item()
            predicted = ((outputs>0.5).int()).float()
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)
    
    # Calculate metrics
    train_loss_epoch = train_loss / len(traindataloader)
    train_acc_epoch = train_correct / train_total
    val_loss_epoch = val_loss / len(validationdataloader)
    val_acc_epoch =  val_correct / val_total
    
    history['train_loss'].append(train_loss_epoch)
    history['train_acc'].append(train_acc_epoch)
    history['val_loss'].append(val_loss_epoch)
    history['val_acc'].append(val_acc_epoch)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}')
    print(f'  Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}')
    print('-' * 50)



end_time = time.time()
execution_time = end_time - start_time    
print("_________________________")
print("    Execution Time - Model Training   ")
print("-------------------------")

hour = execution_time//3600
minutes = (execution_time - 3600*hour)//60
# print(f"Execution time: {execution_time:.4f} seconds")
print(f"{hour:.0f} Hours and {minutes:.1f} Minutes") 


#%%

plt.figure()
plt.title('Loss')
plt.plot(history['train_loss'],color='r',label='Training loss')
plt.plot(history['val_loss'],color='b',label='Validation loss')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.xlim([0,num_epochs])
plt.legend()
plt.savefig("/tmp/Loss.png", dpi=300, bbox_inches='tight')

plt.figure()
plt.title('Accuracy')
plt.plot(history['train_acc'],color='r',label='Training Accuracy')
plt.plot(history['val_acc'],color='b',label='Validation Accuracy')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.xlim([0,num_epochs])
plt.legend()
plt.savefig("/tmp/Accuracy.png", dpi=300, bbox_inches='tight')

#%%
test_predicted = []
test_target = []

with torch.no_grad():
    for data, targets in testingdataloader:
        
        data, targets = data.to(device), targets.to(device)
        
        outputs = model(data)
        outputs = outputs.squeeze(1)
        
        outputs = outputs.float()
        targets = targets.float()
        
        predicted = ((outputs>0.5).int()).float()
        
        test_predicted += (outputs.float()).detach().cpu()
        test_target += targets.detach().cpu()


list_predicted =[]
list_target = []
for test_target_item,test_predicted_item in zip(test_target,test_predicted):
    list_target.append(test_target_item.item())
    list_predicted.append(test_predicted_item.item())

        
#%%
cm = confusion_matrix(np.int16(np.array(list_target)), np.int16(np.array(list_predicted)))
print(f"Confusion Matrix:\n{cm}")

classes = ['normal','infection']

tn, fp, fn, tp = cm.ravel()

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)
disp.plot()
plt.show()
plt.savefig("/tmp/model_cm.png", dpi=300, bbox_inches='tight')


acc = (tp+tn)/(tp+tn+fp+fn)
print(f"Accuracy: {acc:.4f}")

sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
 
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

fpr, tpr, thresholds = roc_curve(np.int16(np.array(list_target)), np.array(list_predicted))
roc_auc = auc(fpr, tpr)

print(f"Original ROC points: {len(fpr)}")
print(f"ROC AUC: {roc_auc}")

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.show()

plt.savefig("/tmp/model_auc.png", dpi=300, bbox_inches='tight')