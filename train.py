import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time
import sys
import argparse

from data import *
from trainer import *

from networks import *


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr_dis', type=float, default=2e-4)
parser.add_argument('--lr_gen', type=float, default=2e-4)
parser.add_argument("--lr_beta", type = tuple, default=(0.5,0.999))
parser.add_argument('--lambda1', type=int, default=1)
parser.add_argument('--lambda2', type=int, default=10)
parser.add_argument('--lambda3', type=int, default=100)
parser.add_argument('--update_epoch_gen', type=int, default=100)
parser.add_argument('--update_epoch_dis', type=int, default=100)
parser.add_argument('--batch_size', type = int , default = 32, help="batch_size of training")
parser.add_argument('--img_size', type = int, default = 128)
parser.add_argument('--n_epochs', type = int, default = 220000)
parser.add_argument('--epoch', type = int, default = 0)
parser.add_argument('--test_batch_size',type = int, default = 16)
parser.add_argument('--n_gpu', type = int, default = 1)
parser.add_argument('--sample_interval', type = int, default = 1)
parser.add_argument('--check_point', type = int, default = 1)
parser.add_argument('--last_epoch', type = int, default = 0)

opts = parser.parse_args()




opts_name = ['batch_size', 'lambda1', 'lambda2', 'lambda3', 'lr_beta',\
     'lr_dis', 'lr_gen', 'update_epoch_dis', 'update_epoch_gen','img_size','n_epochs',\
     'epoch','test_batch_size','n_gpu','sample_interval','check_point','last_epoch']

length_opts = len(opts_name)

hyperparameters = {opts_name[x] : getattr(opts, opts_name[x]) for x in range(length_opts)}


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


#load network
def load_network(network,path,name):
    path_ = "_{}.pth".format(opts.last_epoch)
    path_ = name + path_
    save_path = os.path.join(path,path_)
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():

        namekey = k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network



if opts.epoch  == 0:
    STGAN = STGANtrainer(hyperparameters)
    STGAN.dis = nn.DataParallel(STGAN.dis,device_ids=[1,2,3,4,5])
    STGAN.gen = nn.DataParallel(STGAN.gen,device_ids=[1,2,3,4,5])
    STGAN_dis = STGAN.cuda()
    STGAN_gen = STGAN.cuda()
else:
    STGAN = STGANtrainer(hyperparameters)
    STGAN.dis= load_network(STGAN.dis,"/home2/qianyu_chen/STGAN/discriminator_models",'discriminator')
    STGAN.gen=load_network(STGAN.gen,"/home2/qianyu_chen/STGAN/generator_models","generator")
    print('done')
    STGAN.dis_opt = Adam(STGAN.dis.parameters(), lr = STGAN.hyperparameters['lr_dis'],betas = STGAN.hyperparameters['lr_beta'])
    STGAN.gen_opt = Adam(STGAN.gen.parameters(), lr = STGAN.hyperparameters['lr_gen'],betas = STGAN.hyperparameters['lr_beta'])                
    STGAN.dis = nn.DataParallel(STGAN.dis,device_ids=[1,2])
    STGAN.gen = nn.DataParallel(STGAN.gen,device_ids=[1,2])
    STGAN_dis = STGAN.to(device)
    STGAN_gen = STGAN.to(device)


def make_test_data():
    test_datasets = CelebAdata(path = '/home2/qianyu_chen/STGAN/code/imgs',
                            name_list = test_name_list, 
                            transforms_ = test_transforms)
    test_dataloader = DataLoader(test_datasets,batch_size=hyperparameters['test_batch_size'],
                            shuffle=False)
    return test_dataloader


def _train__(imgs_real, attr_real,epoch,y,x,attr_fake):
    time_s = time.time()
    
    imgs_real = imgs_real.to(device)
                
    attr_real = attr_real.to(device)
    attr_fake = attr_fake.to(device)
    alpha = Variable(torch.randn((imgs_real.size(0), 1, 1, 1)),requires_grad = False).to(device)
    fake = Variable(Tensor((imgs_real.shape[0])).fill_(1.0), requires_grad=False).to(device)
    fake = fake.view(fake.size(0),1)
    imgs_fake, gen_total_loss, recon_loss,attr_loss,adv_loss = STGAN.gen_update(attr_fake, attr_real, imgs_real, epoch)
    adv_loss_D,attr_loss_D,gradients_loss = STGAN.dis_update(imgs_real,attr_fake , attr_real, epoch,alpha,fake)
    #s记录每次迭代所需的时间
    print(y)
    time_e = time.time()
    if isinstance(y,int) :
        sys.stdout.write(
              "\r{} / {}[Epoch {}/{}] [Batch {}/{}] [D adv: {}, attr_loss: {},gradient_loss: {}] [G loss: {}, recon_loss: {}, attr_loss: {},adv_loss: {}] time: {}\r".format(x,y,epoch, opts.n_epochs,\
                        y, len(train_dataloader),\
                        adv_loss_D.item(),attr_loss_D.item(),gradients_loss.item(),\
                        gen_total_loss.item(),recon_loss.item(),\
                        attr_loss.item(),adv_loss.item(),time_e - time_s))



test_dataloader = iter(test_dataloader)


for epoch in range(opts.epoch, opts.n_epochs):
    STGAN.update_lr(epoch)
    STGAN.checkpoint(epoch)
    start = time.time()    
    for y, (imgs_real, attr_real) in enumerate(train_dataloader):
        #make new attribute code
        a = torch.randn((attr_real.size(0),13)) > 0
        b = torch.ones(a.size())
        c = torch.zeros(a.size()) 
        attr_fake = torch.where(a,b,c)
        attr_fake = torch.autograd.Variable(attr_fake, requires_grad = True)
        for x in range(opts.batch_size // 80):
            _train__(imgs_real[x * 80 :(x+1) * 80], attr_real[x * 80:(x+1) * 80],epoch,y,x,attr_fake[x * 80:(x+1) * 80])
        #adv_loss_D,attr_loss_D,gradients_loss = STGAN.dis_update(imgs_real,attr_fake , attr_real, epoch,alpha,fake)
        end = time.time()
        sys.stdout.write("\r\rtime for 2048 images : {}\r\r".format(end - start))
        
    #if hyperparameters['sample_interval'] != -1 and epoch % hyperparameters['sample_interval'] == 0:       
        if y % 10 == 0: 
            with torch.no_grad():    
                try:
                    imgs_real, attr_real = test_dataloader.next()
                except StopIteration:
                    test_dataloader = iter(make_test_data())
                    imgs_real, attr_real = test_dataloader.next()

                a = torch.randn((imgs_real.size(0),13)).to(device) > 0
                b = torch.ones(a.size()).to(device)
                c = torch.zeros(a.size()).to(device)
                attr_fake = torch.autograd.Variable(torch.where(a,b,c).to(device),requires_grad = False) 
                STGAN.sample(y, imgs_real, attr_real, attr_fake)



