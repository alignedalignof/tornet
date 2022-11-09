import os.path
import models
import torch as tr
import torch.nn as nn
import datasets
from time import time, sleep
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
import threading
import queue

qinput = queue.Queue()

def asiof():
    while True:
        try:
            ans = input()
            q = qinput
            if q:
                q.put(ans if ans else "\n")
            else:
                return
        except queue.Full:
            pass

def ainput():
    try:
        return qinput.get_nowait()
    except queue.Empty:
        return None

def sinput(prompt=None):
    if prompt:
        print(prompt)
    return qinput.get()
    
def inspect(model, imgs):
    for img in imgs:
        img = img.clone()
        for y, x in model.nonbg(img):
            img[:, y-5:y+15, x] = tr.tensor([[1], [1], [1]])
            img[:, y, x-15:x+5] = tr.tensor([[1], [1], [1]])
        img = ToPILImage()(img)
        img.show()
        
        ans = sinput("Save? - s\nInspect - i\n")
        if ans == "s":
            return True
        if ans != "i":
            return False
    return False
        
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
    
def train(model, data, epochs, bgs, fgs, mxs):
    loss = nn.CrossEntropyLoss()
    fit = tr.optim.Adam(model.parameters())
    
    PERIOD = 1
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
            
            ans = ainput()
            if time() - start > PERIOD or batch == size - 1 or ans:
                miss, area = 0, 0
                for bg in bgs:
                    area += bg.shape[1]*bg.shape[2]
                    miss += len(model.nonbg(bg, 0.9))
                miss /= area
                
                hit = 0
                for fg in fgs:
                    hit += 1 if len(model.nonbg(fg, 0.9)) else 0
                hit /= len(fgs)
                
                print(f"loss: {cost.item():>10f}  [{epoch}][{batch * len(X):>5d}/{size:>5d}], {hit=:>0.4}, {miss=:>0.4}")
                if hit > 0.997 and miss < 5e-6:
                        return
                    
                ans = sinput("Test - t\nInspect - i\nStop - ENTER") if ans else None
                if ans == "t":
                    test("", model, data)
                    model.train()
                if ans == "i" and inspect(model, mxs):
                    file = f"{model.__class__.__name__}_{int(1000*time())}.net"
                    tr.save(model.state_dict(), file)
                    print("saved to", file)
                if ans == "\n":
                    return
                start = time()

if __name__ == "__main__":
    BGNET_FILE = "bg.net"
    FGNET_FILE = "fg.net"
    
    asio = threading.Thread(daemon=False, target=asiof)
    asio.start()

    data = datasets.TorData(bg_dir="data/bg", fg_dirs=["data/x", "data/need", "data/greed"])
    mixed = datasets.TorData(bg_dir="data/mixed", fg_dirs=[])
    
    bgnet = models.BgNet()
    trainings = DataLoader(datasets.TorBgSet(data, fg=0, counts=[120000, 120000], size=25, stride=59), batch_size=1, shuffle=True)
    train(bgnet, trainings, 1, data.bg, data.fg[0], mixed.bg)
    tr.save(bgnet.state_dict(), BGNET_FILE)
    
    bgnet.load_state_dict(tr.load(BGNET_FILE))
    test("bg", bgnet, DataLoader(datasets.TorBgSet(data, fg=0, counts=[10000, 0], size=25, stride=173), batch_size=64))
    test("fg", bgnet, DataLoader(datasets.TorBgSet(data, fg=0, counts=[0, len(data.fg[0])], size=0, stride=0), batch_size=1))
    
    tornet = models.TorNet()
    fgset = datasets.TorFgSet(data=data, fg=0, counts=[120000, 120000], bgnet=bgnet, size=(25, 25))
    train(tornet, DataLoader(fgset, batch_size=1, shuffle=True), 1, fgset.falsefg, data.fg[0], mixed.bg)
    tr.save(tornet.state_dict(), FGNET_FILE)

    tornet.load_state_dict(tr.load(FGNET_FILE))
    fgset.set_counts([len(fgset.falsefg), 0])
    test("bg", tornet, DataLoader(fgset, batch_size=1))
    fgset.set_counts([0, len(data.fg[0])])
    test("fg", tornet, DataLoader(fgset, batch_size=1))

    images = [bg.clone() for bg in mixed.bg]

    qinput = None
    for img in images:
        wide = bgnet.nonbg(img)
        narrow = tornet.nonbg(img)
        for y, x in wide:
            img[:, y-5:y+10, x] = tr.tensor([[1], [0], [0]])
            img[:, y, x-5:x+10] = tr.tensor([[1], [0], [0]])
        for y, x in narrow:
            img[:, y-10:y+5, x] = tr.tensor([[0], [1], [0]])
            img[:, y, x-10:x+5] = tr.tensor([[0], [1], [0]])
        img = ToPILImage()(img)
        img.show()
        input(">")