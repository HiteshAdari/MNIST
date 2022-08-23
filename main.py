import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    run()

    torch.cuda.init()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = torchvision.datasets.MNIST(root='data/', transform=torchvision.transforms.ToTensor())
    '''
    #Visualizing image
    image, label = dataset[0]
    print('image.shape:', image.shape)
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    print('Label:', label)
    plt.show()
    '''

    val_size = 10000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator(device='cuda'))
    # print(len(train_ds), len(val_ds))

    batch_size = 3000

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              generator=torch.Generator(device='cuda'))
    val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True,
                            generator=torch.Generator(device='cuda'))
    '''
    # Visualizing a random batch of images from train loader
    for images, _ in train_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        plt.show()
        break
    '''


    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)


    train_loader = DeviceDataLoader(train_loader, 'cuda')
    val_loader = DeviceDataLoader(val_loader, 'cuda')

    for images, labels in train_loader:
        print('images.shape:', images.shape)
        inputs = images.reshape(-1, 784)
        print('inputs.shape:', inputs.shape)
        break

    inputs = inputs.to('cuda')
    input_size = inputs.shape[-1]
    hidden_size = 32
    layer1 = nn.Linear(input_size, hidden_size)

    print(inputs.shape)
    layer1_outputs = layer1(inputs)

    print('layer1_outputs.shape:', layer1_outputs.shape)
    layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias
    torch.allclose(layer1_outputs, layer1_outputs_direct, 1e-3)

    relu_outputs = F.relu(layer1_outputs)
    output_size = 10
    layer2 = nn.Linear(hidden_size, output_size)
    layer2_outputs = layer2(relu_outputs)
    print(layer2_outputs.shape)
    F.cross_entropy(layer2_outputs, labels)

    # Expanded version of layer2(F.relu(layer1(inputs)))
    outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias
    torch.allclose(outputs, layer2_outputs, 1e-3)

    # Same as layer2(layer1(inputs))
    outputs2 = (inputs @ layer1.weight.t() + layer1.bias) @ layer2.weight.t() + layer2.bias

    # Create a single layer to replace the two linear layers
    combined_layer = nn.Linear(input_size, output_size)

    combined_layer.weight.data = layer2.weight @ layer1.weight
    combined_layer.bias.data = layer1.bias @ layer2.weight.t() + layer2.bias

    # Same as combined_layer(inputs)
    outputs3 = inputs @ combined_layer.weight.t() + combined_layer.bias


    class MnistModelNN(nn.Module):
        """Feedfoward neural network with 1 hidden layer"""

        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            # hidden layer
            self.linear1 = nn.Linear(in_size, hidden_size)
            # output layer
            self.linear2 = nn.Linear(hidden_size, out_size)

        def forward(self, xb):
            # Flatten the image tensors
            xb = xb.view(xb.size(0), -1)
            # Get intermediate outputs using hidden layer
            out = self.linear1(xb)
            # Apply activation function
            out = F.relu(out)
            # Get predictions using output layer
            out = self.linear2(out)
            return out

        def training_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            return {'val_loss': loss, 'val_acc': acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    input_size = 784
    hidden_size = 32  # you can change this
    num_classes = 10

    model = MnistModelNN(input_size, hidden_size=32, out_size=num_classes)

    for images, labels in train_loader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        print('Loss:', loss.item())
        break

    print('outputs.shape : ', outputs.shape)
    print('Sample outputs :\n', outputs[:2].data)


    def evaluate(model, val_loader):
        """Evaluate the model's performance on the validation set"""
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)


    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        """Train the model using gradient descent"""
        history = []
        optimizer = opt_func(model.parameters(), lr)
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


    model = MnistModelNN(input_size, hidden_size=hidden_size, out_size=num_classes)
    model = model.cuda()
    # to_device(model, 'cuda')

    history = [evaluate(model, val_loader)]
    history += fit(5, 0.5, model, train_loader, val_loader)
    history += fit(10, 0.1, model, train_loader, val_loader)
