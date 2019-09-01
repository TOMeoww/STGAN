import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_,xavier_normal_
from torchsummary import summary


class Conv2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,
                 kernel_size = 4, stride = 2,use_bias = False,
                 gain = 2 ** (0.5), norm = 'BN', pad = (1,1,1,1),
                 activation = 'LR'):
        super(Conv2dBlock,self).__init__()
        
        self.pad = nn.ReflectionPad2d(pad)
        #initialization he_std
        #self.he_std = in_channels * out_channels * kernel_size ** (-0.5) * gain
        #self.weight = nn.Parameter(torch.randn(out_channels,in_channels,kernel_size,kernel_size) * self.he_std)

        #conv and initialization 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias = use_bias)
        kaiming_normal_(self.conv.weight.data)

        if use_bias:
            self.conv.bias.data.zero_()
        
        else:
            pass

        #norm
        if norm == 'BN':
            self.norm = nn.BatchNorm2d(out_channels)
        
        elif norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_channels)
        
        else:
            assert 0,"STGAN's conv block requires IN or BN, not {}".format(norm)


        #activation 
        if activation == 'LR':
            self.activation = nn.LeakyReLU(0.2,inplace = True)

        else:
            assert 0,"STGAN's conv block requires LR, not {}".format(activation)

        self.models = nn.Sequential(self.pad,self.conv,self.norm,self.activation)
    def forward(self,x):


        out = self.models(x)

        return out


class FC(nn.Module):
    def __init__(self,in_channels, out_channels, use_bias = False ,
                 activation = 'LR', gain = 2 ** (0.5)):
        super(FC,self).__init__()
        #he_std
        
        self.he_std = in_channels * (-0.5) * gain
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels ) * self.he_std)
        
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        else:
            self.bias = None         
        #activation
        if activation == 'LR':
            self.activation = nn.LeakyReLU(0.2, inplace = True )
        
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        
        elif activation == None:
            self.activation = None
        
        else:
            assert 0," STGAN's FC reruires LR or Sigmoid, not{}".format(activation)

    def forward(self,x):
        if self.bias is not None:
            out = F.linear( x, self.weight , self.bias )
        
        else:
            out = F.linear( x, self.weight )

        if self.activation:
            out = self.activation( out )
        
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, 
                 stride = 2, padding = 1, use_bias = False, norm = 'BN',
                 activation = 'ReLU'):
        super(DeconvBlock,self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size, stride, bias = use_bias, padding = padding)
        
        if use_bias:
            self.deconv.bias.data.zero_()
        else:
            pass
        
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace = True)
            kaiming_normal_(self.deconv.weight.data)

        elif activation == 'Tanh':
            self.activation = nn.Tanh()
            xavier_normal_(self.deconv.weight.data)
        
        else:
            assert 0," STGAN's FC reruires LR or Tanh, not{}".format(activation) 
        
        #norm
        if norm == 'BN':
            self.norm = nn.BatchNorm2d(out_channels)
            self.models = nn.Sequential(self.deconv,self.norm,self.activation)
        elif norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_channels)
            self.models = nn.Sequential(self.deconv,self.norm,self.activation)
        elif norm == None:
            self.norm = None
            self.models = nn.Sequential(self.deconv,self.activation)
        else:
            assert 0,"STGAN's conv block requires IN or BN, not {}".format(norm)
        
    def forward(self, x):
        out = self.models(x)
        
        return out



class STU(nn.Module):
    def __init__(self, attr_num, in_channels, out_channels, kernel_size=3):
        super(STU,self).__init__()
        self.old_channels = in_channels
        self.new_channels = out_channels
        self.att_num = attr_num
        
        self.deconv = nn.ConvTranspose2d(self.old_channels * 2 + self.att_num , self.new_channels, 
                                         kernel_size = 4, stride = 2, output_padding = 0, padding = 1)
        self.reset_gate = nn.Sequential(nn.Conv2d(self.old_channels + self.old_channels, self.new_channels, kernel_size = 3,
                                                  stride = 1, padding = 1, bias = False),
                                        nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Conv2d( self.old_channels* 2, self.new_channels, kernel_size = 3, 
                                                  stride = 1, padding = 1, bias = False),
                                        nn.Sigmoid())
        self.hidden_gate = nn.Sequential(nn.Conv2d(self.old_channels*2, self.new_channels,
                                                   kernel_size = 3, stride = 1, padding = 1),
                                        nn.Tanh())

    def forward(self, f_enc, state_old, att_target, att_source, encoder_layer_num = 1):
        
        self.batch, _,self.h_old, self.w_old = state_old.size()
        if encoder_layer_num != 5:
            att_diff = (att_target - att_source).view(-1,self.att_num,1,1).expand(self.batch, self.att_num, self.h_old, self.w_old)
            state_hat = self.deconv(torch.cat([state_old, att_diff], dim = 1))
        else:
            state_hat = self.deconv(state_old)

        r = self.reset_gate(torch.cat([f_enc, state_hat], dim = 1))
        z = self.update_gate(torch.cat([f_enc, state_hat], dim = 1))
        state_new = r.mul(z)
        f_new_hat = self.hidden_gate(torch.cat([f_enc, state_new], dim = 1))
        f_new = (1 - z) * state_hat + z.mul(f_new_hat)
        return f_new, state_new


class Generator(nn.Module):
    def __init__(self, n_layers_conv, n_layers_deconv, num_STU, num_att, dim = 64):
        super(Generator,self).__init__()
        self.n_layers_deconv = n_layers_deconv
        self.n_layers_conv = n_layers_conv
        self.num_STU = num_STU
        self.conv_encode = []
        self.attr_num = num_att
        for x in range(n_layers_conv):

            if x == 0:
                
                self.conv_encode.append(Conv2dBlock(3, dim))
            else:
                self.conv_encode.append(Conv2dBlock(dim, dim * 2))
                dim *= 2  
        self.module_list_conv = torch.nn.ModuleList(self.conv_encode)
        self.deconv = []
        for x in range(n_layers_deconv):

            if x == (n_layers_deconv - 1):
                self.deconv.append(DeconvBlock(dim * 3, 3, norm = None, activation = 'Tanh'))
            
            elif x == 0:
                self.deconv.append(DeconvBlock(dim + num_att, dim ))
            
            else:
                self.deconv.append(DeconvBlock(dim * 3, dim ))
            
            dim //= 2 

        dim = 512
        self.module_list_deconv = torch.nn.ModuleList(self.deconv)

        self.STU = []
        for x in range(num_STU):
            self.STU.append(STU(13, dim, dim ))
            dim //= 2
        self.module_list_stu = torch.nn.ModuleList(self.STU)

    def forward(self, x, att_target, att_source):
        for num in range(self.n_layers_conv):

            x = self.module_list_conv[num](x)
            setattr(self, "encode_{}".format(num + 1), x)

        att_diff = (att_target - att_source).view(att_target.size(0),\
                    self.attr_num,1,1).expand(x.size(0), self.attr_num, self.encode_5.size(2), self.encode_5.size(3))
    
        state_list = []
        stu_out_list = []
        f_5 = getattr(self, "encode_{}".format(5))
        
        state_old = torch.cat([f_5, att_diff], dim = 1)

        state_list.append(state_old)

        biaoji_list = [5,1,1,1]
        for num in range(self.num_STU):
            f_enc = getattr(self, "encode_{}".format(4 - num))

            f_new, state_new = self.module_list_stu[num](f_enc, state_list[num ], att_target, att_source,biaoji_list[num])
            state_list.append(state_new)
            stu_out_list.append(f_new)

        out = self.module_list_deconv[0](state_old)
        for num in range(1, self.n_layers_deconv ):
            
            out = torch.cat([out, stu_out_list[num - 1]], dim = 1)
            out = self.module_list_deconv[num](out)
        return out


class Discriminator(nn.Module):
    def __init__(self, conv_num_block,dim,num_att):
        super(Discriminator, self).__init__()
        self.conv_block = []
        self.conv_num_block = conv_num_block
        for x in range(conv_num_block):
            if x == 0:
                self.conv_block.append(Conv2dBlock(3, dim, norm = 'IN'))
            else:
                self.conv_block.append(Conv2dBlock(dim//2, dim, norm = 'IN'))
            
            dim *= 2
        dim //= 2
        self.num_features = dim * 4 * 4 
        self.models_pre = nn.Sequential(*self.conv_block)
        self.dis_att = nn.Sequential(FC(self.num_features, 1024),
                                       FC(1024, num_att,activation = 'Sigmoid'))
        self.dis_adv = nn.Sequential(FC(self.num_features, 1024),
                                     FC(1024, 1, activation = None))

    def forward(self, x):
        
        x = self.models_pre(x).view(x.size(0), -1)
        att = self.dis_att(x)
        adv = self.dis_adv(x)
        return att, adv

