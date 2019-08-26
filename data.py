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
#在data_pre中label_dict 是一个字典比如 label_dict['000004.jpg'] 为 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]即图片对应的标签
#name_list 是每个图片的名字组成的列表
from torch.multiprocessing import Pool, Process, set_start_method

cuda = True if torch.cuda.is_available else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class CelebAdata(Dataset):
    def __init__(self, path, name_list, transforms_=None):
        path_join = partial(os.path.join, path) 
        self.file_path = list(map(path_join, name_list))  
        self.name_list = name_list  
        self.transform = transforms.Compose(transforms_)
    def __getitem__(self,index):
        img = Image.open(self.file_path[index]).convert('RGB')
        attr = Variable(Tensor(labels_dict[self.name_list[index]]),requires_grad = True)
        
        if self.transform is not None:
            img = self.transform(img)
        img = torch.autograd.Variable(img, requires_grad = True)

        return img,attr
    def __len__(self):
        return len(self.file_path)

train_transforms = [transforms.ToTensor()]


test_transforms = [transforms.ToTensor()]


train_datasets = CelebAdata(path = '/home2/qianyu_chen/STGAN/code/imgs',
                            name_list = name_list_train, 
                            transforms_ = train_transforms)
                            
test_datasets = CelebAdata(path = '/home2/qianyu_chen/STGAN/code/imgs',
                            name_list = test_name_list, 
                            transforms_ = test_transforms)


train_dataloader = DataLoader(train_datasets, 
                              batch_size=hyperparameters['batch_size'])


test_dataloader = DataLoader(test_datasets,batch_size=hyperparameters['test_batch_size'],)


