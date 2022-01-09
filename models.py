import torch.nn as nn
import torch.nn.functional as nnf
import torch as tr
import matplotlib.pyplot as plt

class Comb(nn.Module):
    
    def __init__(self, fin, fout, size):
        super().__init__()
        self.filters = nn.ModuleList(nn.ModuleList(nn.Conv2d(1, 1, size) for f in range(fin)) for _ in range(fout))
        self.comb = nn.ModuleList(nn.Conv2d(fin, 1, 1) for _ in range(fout))
        
    def forward(self, x):
        x = [[w(x[:, i, :, :].unsqueeze(1)) for i, w in enumerate(f)] for f in self.filters]
        x = [tr.cat(f, 1) for f in x]
        x = [nnf.relu(f) for f in x]
        x = [c(f) for c, f in zip(self.comb, x)]
        x = tr.cat(x, 1)
        return x
    
class BgNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.r = nn.Conv2d(1, 1, 9)
        self.g = nn.Conv2d(1, 1, 9)
        self.b = nn.Conv2d(1, 1, 9)
        self.conv = nn.Conv2d(3, 1, 1)
        self.lin = nn.Linear(1, 2)
        
    def forward(self, x):
        #plt.imshow(x.squeeze().permute(1, 2, 0))
        #plt.show()
        
        x = [w(x[:, i, :, :].unsqueeze(1)) for i, w in enumerate([self.r, self.g, self.b])]
        x = tr.cat(x, 1)
        x = nnf.relu(x)
        x = self.conv(x)
        x = nnf.max_pool2d(x, (x.shape[2], x.shape[3]))
        x = tr.flatten(x, start_dim=1)
        x = self.lin(x)
        x = tr.sigmoid(x)
        return x
    
    def nonbg(self, img):
        with tr.no_grad():
            img = img.unsqueeze(0)
            img = [w(img[:, i, :, :].unsqueeze(1)) for i, w in enumerate([self.r, self.g, self.b])]
            img = tr.cat(img, 1)
            img = nnf.relu(img)
            img = self.conv(img)
            img = nnf.conv2d(img, self.lin.weight.unsqueeze(-1).unsqueeze(-1), self.lin.bias)
            img = img.argmax(dim=1).squeeze()
            img = img == 1
            return img.nonzero()
        
class TorNet(nn.Module):
    
    def __init__(self, labels):
        super().__init__()
        self.comb = Comb(3, 6, 5)
        self.lin = nn.Linear(16*6, labels)
        
    def forward(self, x):
        #plt.imshow(x.squeeze().permute(1, 2, 0))
        #plt.show()
        
        x = self.comb(x)
        x = self.reduce(x)
        x = tr.flatten(x, start_dim=1)
        x = self.lin(x)
        x = tr.sigmoid(x)
        return x
    
    def reduce(self, x):
        n, c, h, w = x.shape
        window = h//4, w//4
        x = x[:, :, :4*window[0], :4*window[1]]
        x = nnf.max_pool2d(x, window, stride=window)
        return x