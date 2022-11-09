import torch.nn as nn
import torch.nn.functional as nnf
import torch as tr
import matplotlib.pyplot as plt
    
class BgNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.r = nn.Conv2d(1, 1, (4, 8))
        self.g = nn.Conv2d(1, 1, (4, 8))
        self.b = nn.Conv2d(1, 1, (4, 8))
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
    
    def nonbg(self, img, confidence=0.9):
        with tr.no_grad():
            img = img.unsqueeze(0)
            img = [w(img[:, i, :, :].unsqueeze(1)) for i, w in enumerate([self.r, self.g, self.b])]
            img = tr.cat(img, 1)
            img = nnf.relu(img)
            img = self.conv(img)
            img = nnf.conv2d(img, self.lin.weight.unsqueeze(-1).unsqueeze(-1), self.lin.bias)
            img = nnf.sigmoid(img)
            img = img[0,1,:,:] - img[0,0,:,:]
            img = img > confidence
            return img.nonzero()

class TorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fan = nn.ModuleList([nn.Conv2d(1, 1, (4, 8)) for _ in range(3)])
        self.deflate = nn.Conv2d(3, 2, 2, dilation=4, padding=4)
        
    def forward(self, x):
        channels = [x[:,i,:,:].unsqueeze(1) for i in range(x.shape[1])]
        x = [f(c) for c, f in zip(channels, self.fan)]
        x = tr.cat(x, 1)
        x = nnf.relu(x)
        x = self.deflate(x)
        x = nnf.sigmoid(x)
        x = nnf.max_pool2d(x, kernel_size=x.shape[2:])
        x = tr.flatten(x, start_dim=1)
        return x
    
    def nonbg(self, img, confidence=0.9):
        with tr.no_grad():
            img = img.unsqueeze(0)
            channels = [img[:,i,:,:].unsqueeze(1) for i in range(img.shape[1])]
            img = [f(c) for c, f in zip(channels, self.fan)]
            img = tr.cat(img, 1)
            img = nnf.relu(img)
            img = self.deflate(img)
            img = nnf.sigmoid(img)
            img = img[0,1,:,:] - img[0,0,:,:]
            img = img > confidence
            return img.nonzero()