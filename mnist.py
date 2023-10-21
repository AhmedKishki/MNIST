# import dependencies
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# get data
train = MNIST(root="Datasets", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instance of NN, loss, optimizer
clf = ImageClassifier()
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(10):
        for (X, y) in dataset:
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            
            # backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
    
        print(f"epoch: {epoch}, loss: {loss.item()}")
    
    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)