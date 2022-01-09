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
    
    def __init__(self, data, counts, size, stride):
        self.counts = counts
        self.cdf = [sum(counts[:i+1]) for i in range(len(counts))]
        self.data = data
        self.size, self.stride = size, stride
        
    def __len__(self):
        return self.cdf[-1]
    
    def __getitem__(self, index):
        label = sum([index >= m for m in self.cdf])
        index = index % self.counts[label]
        if label == 0:
            return self.data.get_bg(index, self.size, self.stride), tr.tensor([1., 0.])
        fg = label - 1
        return self.data.get_fg(fg, index), tr.tensor([0., 1.])

class TorFgSet(Dataset):

    def __init__(self, data, counts, sizes, stride, bgnet):
        self.set_counts(counts)
        self.data = data
        self.sizes = sizes
        self.labels = data.fgcount + 1
        self.bg = [[] for _ in sizes]
        bgnet.eval()
        
        def overlap(hit, x, y, span):
            mask = tr.BoolTensor(len(hit))
            for i, h in enumerate(hit):
                tx = h[1].item()
                ty = h[0].item()
                mask[i] = tx >= x - span and tx < x + span and ty >= y - span and ty < y + span
            return mask

        with tr.no_grad():
            for bg in data.bg:
                for size, slot in zip(self.sizes, self.bg):
                    points = {}
                    fg = bgnet.nonbg(bg)
                    for y, x in fg:
                        y = fg[0, 0].item()
                        x = fg[0, 1].item()
                        sub = bg[:, max(0, y - size):(y + size), max(0, x - size):x + size]
                        sub = sub.unfold(1, size, stride).unfold(2, size, stride)
                        for h in range(sub.shape[1]):
                            for w in range(sub.shape[2]):
                                if (y + w, x + h) in points:
                                    continue
                                points[(y + w, x + h)] = 1
                                slot.append(sub[:,h,w, :,:].squeeze())
    
    def set_counts(self, counts):
        self.counts = counts
        self.cdf = [sum(counts[:i+1]) for i in range(len(counts))]
        
    def __len__(self):
        return self.cdf[-1]
    
    def __getitem__(self, index):
        label = sum(index >= m for m in self.cdf)
        index = index % self.counts[label]
        onehot = tr.zeros(len(self.counts))
        onehot[label] = 1
        if label == 0:
            slot = index % len(self.sizes)
            index = (index // len(self.sizes)) % len(self.bg[slot])
            return self.bg[slot][index], onehot
        fg = label- 1
        return self.data.get_fg(fg, index), onehot
    