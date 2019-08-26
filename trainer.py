import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
import time
from networks import Generator,Discriminator
import numpy as np

#self.hyperparameters是一个包含部分训练所需超参数的字典

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BCE = torch.nn.BCELoss().cuda()


#update learning_rate according to epoch
def adjust_lr(epoch, optimizer):
    if epoch == 100:
        lr = 2e-5
        for params_group in optimizer.param_groups:
            params_group['lr'] = lr




class STGANtrainer(nn.Module):
    def __init__(self, hyperparameters):
        super(STGANtrainer, self).__init__()
        self.hyperparameters = hyperparameters
        self.gen = Generator(5,5,4,13)
        self.dis = Discriminator(5,64,13) 
        
        self.dis_opt = Adam(self.dis.parameters(), lr = self.hyperparameters['lr_dis'],betas = self.hyperparameters['lr_beta'])
        self.gen_opt = Adam(self.gen.parameters(), lr = self.hyperparameters['lr_gen'],betas = self.hyperparameters['lr_beta'])                
        self.dis_attr_opt = Adam(self.gen.parameters(), lr = 0.5 * self.hyperparameters['lr_gen'],betas = self.hyperparameters['lr_beta'])

    def compute_gradient_penalty(self, fake_samples, real_samples,alpha,fake):

        samples = (fake_samples.detach() * alpha + real_samples * (1 - alpha)).requires_grad_()
        _, score_adv = self.dis(samples)
        
        gradients = torch.autograd.grad(outputs=score_adv,
                                        inputs=samples,
                                        grad_outputs=fake,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True,)[0]
        gradients = gradients.view(score_adv.size(0), -1)
        gradients = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
        return gradients

    
    def compute_attr_loss(self, attr, attr_target, mode = 'gen'):
        attr_target = torch.autograd.Variable(attr_target,requires_grad = False)
        if mode == 'gen':
            return BCE(attr, attr_target)
        
        elif mode == 'dis': 
            return BCE(attr, attr_target)
        
        else:
            assert 0, "mode for attr loss must be gen or dis not {}".format(mode)



    def compute_adv_loss(self, fake_validity = None, real_validity = None, mode = 'gen'):
        if mode == 'dis':         
            return -torch.mean(real_validity) + torch.mean(fake_validity)
        
        elif mode == 'gen':

            return -torch.mean(fake_validity)
        
        else:
            assert 0, "mode for adv loss must be gen or dis not {}".format(mode)


    def compute_recon_loss(self, imgs_fake, imgs_pair):
        imgs_pair = torch.autograd.Variable(imgs_pair,requires_grad = False)
        loss = F.l1_loss(imgs_fake, imgs_pair)
        imgs_pair.requires_grad_()        
        return loss
    

    def gen_update(self, attr_fake, attr_real, imgs_real, epoch):
        self.gen_opt.zero_grad()
        #recon loss
        imgs_recon = self.gen(imgs_real, attr_real, attr_real)
        recon_loss = self.compute_recon_loss(imgs_recon,imgs_real)

        #attr_loss
        imgs_fake = self.gen(imgs_real, attr_fake, attr_real)
        attr, adv = self.dis(imgs_fake)
        attr_loss = self.compute_attr_loss(attr, attr_fake, mode = 'gen')

        #adv loss
        adv_loss = 0.000000050*self.compute_adv_loss(adv, mode = 'gen')

        gen_total_loss = recon_loss * 10 * self.hyperparameters['lambda3'] +  attr_loss * self.hyperparameters['lambda2'] + adv_loss
        
        gen_total_loss.backward(retain_graph=True)

        self.gen_opt.step()  
        self.gen_opt.zero_grad()
        imgs_fake = self.gen(imgs_real, attr_fake, attr_real)
        attr, _ = self.dis(imgs_fake)
        attr_loss = 5*self.compute_attr_loss(attr, attr_fake, mode = 'gen')                         
        self.gen_opt.step()                                          
        return imgs_fake,gen_total_loss,recon_loss* self.hyperparameters['lambda3'],2*attr_loss,0.0000000010 *adv_loss


    def dis_update(self, imgs_real,attr_fake, attr_target_real, epoch, alpha, fake):
        self.dis_opt.zero_grad()
        imgs_fake = self.gen(imgs_real,attr_fake,attr_target_real).detach_()
        _, fake_validity = self.dis(imgs_fake)
        
        attr, real_validity = self.dis(imgs_real)
        #adv_loss
        adv_loss = 0.0000050*self.compute_adv_loss(fake_validity, real_validity, mode = 'dis')
        attr_loss = 10*self.hyperparameters['lambda1'] * self.compute_attr_loss(attr, attr_target_real, mode = 'dis')
        gradients_loss = self.compute_gradient_penalty(imgs_fake,imgs_real,alpha,fake)
                                      
        d_loss = gradients_loss +  adv_loss + attr_loss 
        d_loss.backward(retain_graph=True)

        self.dis_opt.step()
        self.dis_attr_opt.zero_grad()
        attr, real_validity = self.dis(imgs_real)
        attr_loss = 10*self.compute_attr_loss(attr, attr_target_real, mode = 'dis')
        attr_loss.backward(retain_graph = True)
        self.dis_attr_opt.step()
        return adv_loss,attr_loss,gradients_loss

    def forward(self, imgs_real, attr_real, attr_fake):
        
        imgs_fake = self.gen(imgs_real, attr_fake, attr_real)
        imgs_recon = self.gen(imgs_real, attr_real, attr_real)
        
        return imgs_fake, imgs_recon
    
    def save_model(self,epoch):
        torch.save(self.gen.state_dict(), "/home2/qianyu_chen/STGAN/generator_models/generator_%d.pth" % epoch)
        torch.save(self.dis.state_dict(), "/home2/qianyu_chen/STGAN/discriminator_models/discriminator_%d.pth" % epoch)
        torch.save(self.gen, "/home2/qianyu_chen/STGAN/generator_models/gen_%d.pth" % epoch)
        torch.save(self.dis, "/home2/qianyu_chen/STGAN/discriminator_models/dis_%d.pth" % epoch)
    def sample(self,epoch, imgs_real, attr_real, attr_fake):
        imgs_fake, imgs_recon = self.forward(imgs_real, attr_real, attr_fake)
        #imgs_fake = (imgs_fake + 1.0) * 0.5 
        #imgs_recon = (imgs_recon + 1.0) * 0.5 
        save_image(imgs_fake, filename = '/home2/qianyu_chen/STGAN/code/sample_images_fake/{}.jpg'.format(epoch), nrow = 4)
        save_image(imgs_recon, filename = '/home2/qianyu_chen/STGAN/code/sample_imgs_recon/{}.jpg'.format(epoch), nrow = 4)
        
    def checkpoint(self,epoch):
        if self.hyperparameters['check_point'] != -1 and epoch % self.hyperparameters['check_point'] == 0:
            self.save_model(epoch)

    def update_lr(self,epoch):
        if epoch == self.hyperparameters['update_epoch_dis']:
            adjust_lr(epoch, self.dis_opt)
            adjust_lr(epoch, self.gen_opt)
