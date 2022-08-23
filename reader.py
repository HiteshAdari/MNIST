import torch
from linear import *
import matplotlib.pyplot as plt

testdat = torchvision.datasets.MNIST('data', train=False, transform=torchvision.transforms.ToTensor())

result0 = evaluate(model, val_loader)
print('result0:', result0)

history = fit(20, 0.001, model, train_loader, val_loader)


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


test_loader = DataLoader(testdat, batch_size=256)
result = evaluate(model, test_loader)
print(result)

torch.save(model.state_dict(), 'mnist-logistic.pth')
model2 = MnistModel()
evaluate(model2, testdat)

model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()
test_loader = DataLoader(testdat, batch_size=256)
result = evaluate(model2, test_loader)
print(result)
