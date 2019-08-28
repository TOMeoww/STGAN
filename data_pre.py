import glob
import os
import numpy as np
import collections
from functools import partial
from PIL import Image
import torch
print('begin!')
#提取出13个属性的label
with open('/home2/qianyu_chen/CelebA/Anno/list_attr_celeba.txt', 'rt') as f:
    a = f.readlines()

attr_index = [0,4,5,8,9,11,12,15,20,21,22,24,26,39]
attr_index = [x + 1 for x in attr_index]
attr_index[0] = attr_index[0] - 1
b = []
for x in a:
    b.append(x.strip().split())
attr = b

#将标签中的-1改为0
for x in range(len(attr)):
    for y in range(41):
        if attr[x][y] == '1':
            attr[x][y] = 1
        elif attr[x][y] == '-1':
            attr[x][y] = 0

attr_required = []

for x in attr:
    attr_required.append([x[y] for y in attr_index])

labels_dict = {x[0] : x[1:] for x in attr_required}

name_list = []

for x in os.walk('/home2/qianyu_chen/STGAN/code/imgs'):
    path_root = x[0]
    name_list = list(*x[2:])

test_name_list = name_list[:1000]
name_list_train = name_list[1000:]

def load_image_PIL(path):
    path = os.path.join('/home2/qianyu_chen/STGAN/code/imgs',path)
    img = Image.open(path).convert('RGB')
    img = torch.from_numpy((np.array(img) / 127.5 - 1).transpose((2, 0, 1)))
    return img

path_join = partial(os.path.join, '/home2/qianyu_chen/STGAN/code/imgs')

test_path = list(map(path_join, test_name_list))
train_path = list(map(path_join, name_list_train))

#make train images and attr list
train_data = {x : [load_image_PIL(x),torch.Tensor(labels_dict[x])] for x in name_list_train}

train_images = torch.stack([train_data[x][0] for x in name_list_train])
train_attr = torch.stack([train_data[x][1] for x in name_list_train])
del train_data

#make test data and attr
test_data = {x : [load_image_PIL(x), torch.Tensor(labels_dict[x])] for x in test_name_list}

test_images = torch.stack([test_data[x][0] for x in test_name_list])
test_attr = torch.stack([test_data[x][1] for x in test_name_list])
del test_data
print('load_numpy_image_done!')import glob
import os
import numpy as np
import collections
from functools import partial
from PIL import Image
import torch
print('begin!')
#提取出13个属性的label
with open('/home2/qianyu_chen/CelebA/Anno/list_attr_celeba.txt', 'rt') as f:
    a = f.readlines()

attr_index = [0,4,5,8,9,11,12,15,20,21,22,24,26,39]
attr_index = [x + 1 for x in attr_index]
attr_index[0] = attr_index[0] - 1
b = []
for x in a:
    b.append(x.strip().split())
attr = b

#将标签中的-1改为0
for x in range(len(attr)):
    for y in range(41):
        if attr[x][y] == '1':
            attr[x][y] = 1
        elif attr[x][y] == '-1':
            attr[x][y] = 0

attr_required = []

for x in attr:
    attr_required.append([x[y] for y in attr_index])

labels_dict = {x[0] : x[1:] for x in attr_required}

name_list = []

for x in os.walk('/home2/qianyu_chen/STGAN/code/imgs'):
    path_root = x[0]
    name_list = list(*x[2:])

test_name_list = name_list[:1000]
name_list_train = name_list[1000:]

def load_image_PIL(path):
    path = os.path.join('/home2/qianyu_chen/STGAN/code/imgs',path)
    img = Image.open(path).convert('RGB')
    img = torch.from_numpy((np.array(img) / 127.5 - 1).transpose((2, 0, 1)))
    return img

path_join = partial(os.path.join, '/home2/qianyu_chen/STGAN/code/imgs')

test_path = list(map(path_join, test_name_list))
train_path = list(map(path_join, name_list_train))

#make train images and attr list
train_data = {x : [load_image_PIL(x),torch.Tensor(labels_dict[x])] for x in name_list_train}

train_images = torch.stack([train_data[x][0] for x in name_list_train])
train_attr = torch.stack([train_data[x][1] for x in name_list_train])
del train_data

#make test data and attr
test_data = {x : [load_image_PIL(x), torch.Tensor(labels_dict[x])] for x in test_name_list}

test_images = torch.stack([test_data[x][0] for x in test_name_list])
test_attr = torch.stack([test_data[x][1] for x in test_name_list])
del test_data
print('load_numpy_image_done!')
