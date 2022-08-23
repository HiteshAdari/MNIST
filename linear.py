
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = "cpu"

traindat = torchvision.datasets.MNIST('data', train=True, transform=torchvision.transforms.ToTensor())


# splitting for training and cross-validation, using random as it maybe sorted 0-10
X, Xval = random_split(traindat, [55000, 5000])

batch_size = 128
train_loader = DataLoader(X, batch_size, shuffle=True)
val_loader = DataLoader(Xval, batch_size)

input_size = 28 * 28
num_classes = 10


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


model = MnistModel().to(device)

for images, labels in train_loader:
    #print(images.shape)
    outputs = model(images)
    break

# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)
max_probs, preds = torch.max(probs, dim=1)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


loss_fn = F.cross_entropy

# Loss for current batch of data
loss = loss_fn(outputs, labels)



def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results

    for epoch in range(epochs):

        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history
