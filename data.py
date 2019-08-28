import glob
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from functools import partial
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
from hyper import *
from data_pre import *
'''在data_pre中label_dict 是一个字典比如 label_dict['000004.jpg'] 为 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]即图片对应的标签
name_list 是每个图片的名字组成的列表
'''


cuda = True if torch.cuda.is_available else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CelebAdata(Dataset):
    def __init__(self, data_array, attr_array, batch_size):  
        self.data_array = data_array  
        self.attr_array = attr_array
        self.batch_size = batch_size
        self.count = 0
    def __len__(self):
        return len(self.attr_array // self.batch_size)
    
    def __iter__(self):
        return self

    def __next__(self):
        data = self.data_array[self.count * self.batch_size : (self.count + 1) * self.batch_size]
        attr = self.attr_array[self.count * self.batch_size : (self.count + 1) * self.batch_size]
        self.count += 1
        data = Variable(data.to(device, dtype=torch.float),requires_grad = True)
        attr = Variable(attr.to(device, dtype=torch.float),requires_grad = True)
        return data, attr  
    
    
