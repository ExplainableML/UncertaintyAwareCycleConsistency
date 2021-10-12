import torch.utils.data as data
import os.path
import numpy as np
import torch

class Images_w_nameList(data.Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, root1, root2, flist1, flist2, transform=None):
        self.root1 = root1
        self.root2 = root2
        self.flist1 = flist1
        self.flist2 = flist2
        self.transform = transform
    def __getitem__(self, index):
        impath1 = self.flist1[index]
        img1 = np.load(os.path.join(self.root1, impath1))
        impath2 = self.flist2[index]
        img2 = np.load(os.path.join(self.root2, impath2))
        img1 = torch.tensor(img1)
        img2 = torch.tensor(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2
    def __len__(self):
        return len(self.flist1)