from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import torch as tr
import matplotlib.pyplot as plt

class TorData:
    
    def __init__(self, bg_dir, fg_dirs):
        self.bgpngs = [png for png in glob(f"{bg_dir}/*.png")]
        self.bg = [ToTensor()(Image.open(png).convert("RGB")) for png in self.bgpngs]
        self.fg = [[ToTensor()(Image.open(png).convert("RGB")) for png in glob(f"{fg}/*.png")] for fg in fg_dirs]
        self.bgcount = len(self.bg)
        self.fgcount = len(self.fg)
        
    def get_fg(self, fg, i):
        return self.fg[fg][i % len(self.fg[fg])]
    
    def get_bg(self, i, size, stride):
        bg = self.bg[i % self.bgcount]
        i = i // self.bgcount
        c, h, w = bg.shape
        h, w = h - size, w - size
        n = i * stride
        y = n // w
        y = (y * stride) % h
        x = n % w
        return bg[:, y:y+size, x:x+size]

class TorBgSet(Dataset):
    
    def __init__(self, data, fg, counts, size, stride):
        self.counts = counts
        self.data = data
        self.fg = fg
        self.size, self.stride = size, stride
        
    def __len__(self):
        return sum(self.counts)
    
    def __getitem__(self, index):
        label = int(index >= self.counts[0])
        index = index - sum(self.counts[:label])
        if label:
            #x = self.data.get_fg(self.fg, index)
            #plt.imshow(x.squeeze().permute(1, 2, 0))
            #plt.show()
            return self.data.get_fg(self.fg, index), tr.tensor([0., 1.])
        return self.data.get_bg(index, self.size, self.stride), tr.tensor([1., 0.])


class TorFgSet(Dataset):

    def __init__(self, data, fg, counts, bgnet, size):
        self.set_counts(counts)
        self.data = data
        self.falsefg = []
        self.fg = fg
        bgnet.eval()
        
        h, w = size
        for bg in data.bg:
            for y, x in bgnet.nonbg(bg, 0.01):
                for dx in [-5, -2, 0, 2, 5]:
                    for dy in [-5, -2, 0, 2, 5]:
                        i, j = y + dy, x + dx
                        falsefg = bg[:,i:i+h, j:j+w]
                        if falsefg.shape[1] == h and falsefg.shape[2] == w:
                            self.falsefg.append(falsefg)
    
    def set_counts(self, counts):
        self.counts = counts
        
    def __len__(self):
        return sum(self.counts)
    
    def __getitem__(self, index):
        label = int(index >= self.counts[0])
        index = index - sum(self.counts[:label])
        if label:
            return self.data.get_fg(self.fg, index), tr.tensor([0., 1.])
        return self.falsefg[index % len(self.falsefg)], tr.tensor([1., 0.])
    