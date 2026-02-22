import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from modelmps_ import SimpleCNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=64)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start = time.time()
for epoch in range(10):
    model.train()
    correct, total = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        correct += out.argmax(1).eq(target).sum().item()
        total += target.size(0)
    train_acc = 100. * correct / total

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
    test_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/10 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print(f"Done in {time.time()-start:.1f}s")
torch.save(model.state_dict(), 'simple_mnist.pth')
print("Saved to simple_mnist.pth")