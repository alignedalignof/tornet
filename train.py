import os.path
import models
import torch as tr
import torch.nn as nn
import datasets
from time import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test(desc, model, data):
    loss = nn.CrossEntropyLoss()
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with tr.no_grad():
        for X, y in data:
            pred = model(X)
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(1) == y.argmax()).type(tr.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{model.__class__.__name__} {desc} test[{size}]: accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}")
    
def train(model, data, epochs):
    loss = nn.CrossEntropyLoss()
    fit = tr.optim.Adam(model.parameters())
    
    PERIOD = 5
    start = time() - PERIOD
    size = len(data.dataset)
    model.train()
    for epoch in range(epochs):
        for batch, (X, Y) in enumerate(data):
            Yh = model(X)
            cost = loss(Yh, Y)
            
            fit.zero_grad()
            cost.backward()
            fit.step()
            
            if time() - start > PERIOD or batch == size - 1:
                start = time()
                print(f"loss: {cost.item():>7f}  [{epoch}][{batch * len(X):>5d}/{size:>5d}]")

def fg_label(tornet, img, x, y):
    sizes = 20, 12
    for size in sizes:
        for j in range(max(0, y - size), y + size):
            for i in range(max(0, x - size), x + size):
                sub = img[:, j:j+size, i:i+size]
                if sub.shape[1] < size or sub.shape[2] < size:
                    continue
                label = tornet(sub.unsqueeze(0)).argmax()
                if label != 0:
                    return label, j, i, size
    return 0, y, x, 1
    
if __name__ == "__main__":
    BGNET_FILE = "bgnet.tch"
    FGNET_FILE = "tornet.tch"
    
    data = datasets.TorData(bg_dir="data/bg", fg_dirs=["data/x", "data/need", "data/greed"])
    
    bgnet = models.BgNet()
    trainings = DataLoader(datasets.TorBgSet(data, counts=[60600, 20000, 20000, 20000], size=25, stride=59), batch_size=1, shuffle=True)
    train(bgnet, trainings, 1)
    tr.save(bgnet.state_dict(), BGNET_FILE)
    
    bgnet.load_state_dict(tr.load(BGNET_FILE))
    test("background", bgnet, DataLoader(datasets.TorBgSet(data, counts=[10000, 0, 0, 0], size=25, stride=173), batch_size=64))
    for fg in 0, 1, 2:
        counts=[(len(data.fg[fg]) if fg == i else 0) for i in (-1, 0, 1, 2)]
        test(f"label {fg}", bgnet, DataLoader(datasets.TorBgSet(data, counts=counts, size=0, stride=0), batch_size=1))
    
    tornet = models.TorNet(2)
    fgset = datasets.TorFgSet(data, counts=[33000, 30000], sizes=(20, 12), stride=1, bgnet=bgnet)
    train(tornet, DataLoader(fgset, batch_size=1, shuffle=True), 1)
    tr.save(tornet.state_dict(), FGNET_FILE)
    
    tornet.load_state_dict(tr.load(FGNET_FILE))
    fgset.set_counts([sum(len(bg) for bg in fgset.bg), 0])
    test("background", tornet, DataLoader(fgset, batch_size=1))
    for fg in 0,:
        fgset.set_counts([(len(data.fg[fg]) if fg == i else 0) for i in (-1, 0)])
        test(f"label {fg}", tornet, DataLoader(fgset, batch_size=1))
    
    mixed = datasets.TorData(bg_dir="data/mixed", fg_dirs=[])
    images = [bg.clone() for bg in mixed.bg]
    colors = [tr.tensor([[1], [0], [0]]), tr.tensor([[1], [1], [1]]), tr.tensor([[1], [1], [0]]), tr.tensor([[0], [1], [1]])]
    for img in images:
        hits = []
        for y, x in bgnet.nonbg(img):
            label, y, x, size = fg_label(tornet, img, x, y)
            hits.append((colors[label], y, x, size))
            print(f"fg {label} @ {x}, {y} [{size}]")
        for c, y, x, s in hits:
            img[:, y, x:x+s] = c
            img[:, y+s, x:x+s] = c
            img[:, y:y+s, x] = c
            img[:, y:y+s, x+s] = c
        plt.imshow(img.permute(1, 2, 0))
        plt.show(block=True)
