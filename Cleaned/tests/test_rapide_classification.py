import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
# %matplotlib inline
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn

## MNIST dataset (images and labels)
dataset = MNIST(root='data/', download=True)
print(len(dataset))

image, label = dataset[10]
plt.imshow(image, cmap='gray')
print('Label:', label)

mnist_dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())
print(mnist_dataset)

image_tensor, label = mnist_dataset[0]
print(image_tensor.shape, label)

## Plot the image of the tensor
plt.imshow(image_tensor[0, 10:15, 10:15], cmap='gray')

train_data, validation_data = random_split(mnist_dataset, [50000, 10000])
print("Length of Train Datasets:", len(train_data))
print("Length of Validation Datasets:", len(validation_data))

## Model Definition
input_size = 28 * 28
num_classes = 10

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate the loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

## Helper Functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation Phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

## DataLoaders
batch_size = 128
train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size, shuffle=False)

## Model Instance
model = MnistModel()

## Evaluation and Training
result0 = evaluate(model, val_loader)
history1 = fit(5, 0.001, model, train_loader, val_loader)