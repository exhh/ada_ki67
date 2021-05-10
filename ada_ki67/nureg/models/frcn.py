import numpy as np
import torch
import torch.nn as nn
from ..torch_utils import to_device
from ..torch_utils import Indexflow
from .models import register_model
import torch.nn.functional as F
import functools

__all__ = ['frcn', 'FRCN']

class FRCN_Passthrough(nn.Module):
    def __init__(self, **kwargs):
        super(FRCN_Passthrough, self).__init__()
    def forward(self, x, **kwargs):
        return x

def match_tensor(out, refer_shape):
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0))
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]

    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row))
    else:
        crop_row = row - skiprow
        left_crop_row  = crop_row // 2

        right_row = left_crop_row + skiprow

        out = out[:,:,left_crop_row:right_row, :]

    return out

def get_activ(activ=True):
    if activ is 'elu':
        return nn.ELU(inplace=True, alpha=1)
    elif activ is 'selu':
        return nn.SELU(inplace=True)
    elif activ is 'relu':
        return nn.ReLU(inplace=True)
    elif activ is 'linear':
        return FRCN_Passthrough()
    else:
        return nn.PReLU()

def getNormLayer(norm='bn', dim=2):
    norm_layer = None
    if dim == 2:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        # elif norm == 'ln':
        #     norm_layer = functools.partial(LayerNorm2d)
    elif dim == 1:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm1d, affine=False)
        # elif norm == 'ln':
        #     norm_layer = functools.partial(LayerNorm1d)
    assert(norm_layer != None)
    return norm_layer

def conv_norm(inChans, outChans, filters = 64, kernel_size=(3,3), padding=1,
              activ='relu', conv_activ = 'linear', norm = 'bn'):
    _layers = []
    _layers += [getNormLayer(norm)(inChans )]
    _layers += [get_activ(activ)]
    _layers += [nn.Conv2d(inChans, outChans, kernel_size=kernel_size, padding=padding)]
    _layers += [get_activ(conv_activ)]
    return nn.Sequential(*_layers)

class FRCN_UpConv(nn.Module):
    def __init__(self, inChans, outChans,  kernel_size = 3, padding=1,
                 activ='relu', conv_activ = 'linear', dest_size=None):
        super(FRCN_UpConv, self).__init__()
        #hidChans = outChans // 2

        self.up_layer = nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv_bn  = conv_norm(inChans, outChans, kernel_size=kernel_size, padding=padding,
                                  activ=activ, conv_activ = conv_activ)
    def forward(self, x, dest_size):
        '''
        dest_size should be (row, col)
        '''
        out = match_tensor(self.up_layer(x), dest_size).contiguous()
        out = self.conv_bn(out)
        return out

class FRCN_ResBlock(nn.Module):
    def __init__(self, nChans, kernel_size=3, scaling=1, activ= 'elu', norm='bn'):
        super(FRCN_ResBlock, self).__init__()
        #hidChans = outChans // 2
        self.droprate = 0.33
        self.scaling = scaling

        _layers = []
        _layers += [getNormLayer(norm)(nChans)]
        _layers += [get_activ(activ)]
        _layers += [nn.Conv2d(nChans, nChans, kernel_size=kernel_size, padding=1)]
        _layers += [nn.Dropout2d(self.droprate)]
        _layers += [getNormLayer(norm)(nChans )]
        _layers += [get_activ(activ)]
        _layers += [nn.Conv2d(nChans, nChans, kernel_size=kernel_size, padding=1)]

        self.res_path = nn.Sequential(*_layers)

    def forward(self, x):
        '''
        dest_size should be (row, col)
        '''
        res = self.res_path(x)
        out = self.scaling * res + x
        return out

class FRCN(nn.Module):
    # Implementation for the MIA paper
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, img_channels=3, scaling=0.3, norm='bn', activ='elu', last_activ='relu', nf=32):
        super(FRCN, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))

        _layers = []
        _layers += [conv_norm(img_channels, nf, kernel_size=(1,1), padding=0, activ=activ, norm=norm)]
        _layers += [FRCN_ResBlock(nf,   scaling=scaling, activ=activ)]
        self.block1 = nn.Sequential(*_layers)

        _layers = [conv_norm(nf, nf*2, kernel_size=(1,1), padding=0, activ=activ, norm=norm)]
        _layers += [FRCN_ResBlock(nf*2,   scaling=scaling, activ=activ)]
        _layers += [nn.MaxPool2d(2)]
        _layers += [FRCN_ResBlock(nf*2,   scaling=scaling, activ=activ)]
        self.block2 = nn.Sequential(*_layers)

        _layers = [conv_norm(nf*2, nf*4, kernel_size=(1,1), padding=0, activ=activ, norm=norm)]
        _layers += [nn.MaxPool2d(2)]
        _layers += [FRCN_ResBlock(nf*4,   scaling=scaling, activ=activ)]
        self.block3 = nn.Sequential(*_layers)

        _layers = [conv_norm(nf*4, nf*8, kernel_size=(1,1), padding=0, activ=activ, norm=norm)]
        _layers += [nn.MaxPool2d(2)]
        _layers += [FRCN_ResBlock(nf*8,   scaling=scaling, activ=activ)]
        self.block4 = nn.Sequential(*_layers)

        _layers = [nn.MaxPool2d(2)]
        _layers += [FRCN_ResBlock(nf*8,   scaling=scaling, activ=activ)]
        self.block5 = nn.Sequential(*_layers)
        self.up_1 = FRCN_UpConv(nf*8, nf*8, kernel_size = 1, padding=0)

        self.pre_2 =  FRCN_ResBlock(nf*8,   scaling=scaling, activ=activ)
        self.up_2 = FRCN_UpConv(nf*8, nf*4, kernel_size = 1, padding=0)

        self.pre_3 = FRCN_ResBlock(nf*4,   scaling=scaling, activ=activ)
        self.up_3  = FRCN_UpConv(nf*4, nf*2, kernel_size = 1, padding=0)

        self.pre_4 = FRCN_ResBlock(nf*2,   scaling=scaling, activ=activ)
        self.up_4  = FRCN_UpConv(nf*2, nf, kernel_size = 1, padding=0)

        _layers =  [FRCN_ResBlock(nf,   scaling=scaling, activ=activ)]
        _layers += [conv_norm(nf, 1, kernel_size=(1,1), padding=0, activ=activ, conv_activ=last_activ, norm=norm)]
        self.last_conv = nn.Sequential(*_layers)

    def forward(self, x, adv=True):
        x = to_device(x,self.device_id)
        x = self.block1(x)
        size_1 = x.size()[2:]

        x = self.block2(x)
        size_2 = x.size()[2:]

        x = self.block3(x)
        size_3 = x.size()[2:]

        x = self.block4(x)
        size_4 = x.size()[2:]

        x = self.block5(x)
        x = self.up_1(x, size_4)

        x = self.pre_2(x)
        x = self.up_2(x, size_3)

        x = self.pre_3(x)
        x = self.up_3(x, size_2)

        x = self.pre_4(x)
        x = self.up_4(x, size_1)

        out = self.last_conv(x)
        return out

    def predict(self, x, batch_size=None):
        self.eval()
        total_num = x.shape[0]
        if batch_size is None or batch_size >= total_num:
            x = to_device(x, self.device_id).float()
            det= self.forward(x, adv=False)
            return det.cpu().data.numpy()
        else:
            det_results = []
            #advs = []
            for ind in Indexflow(total_num, batch_size, False):
                #devInd = to_device(torch.from_numpy(ind), self.device_id, False)
                #Ind = torch.from_numpy(ind)
                data = x[ind]
                data = to_device(data, self.device_id, False).float()

                det = self.forward(data)
                det_results.append(det.cpu().data.numpy())
                #advs.append(adv.cpu().data.numpy())
            return np.concatenate(det_results,axis=0)

@register_model('frcn')
def frcn(pretrained=True, finetune=False, out_map=True, **kwargs):
    model = FRCN()
    return model
