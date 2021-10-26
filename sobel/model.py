import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image

def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    # print('1')
    # print(kernel.dtype)
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        """
        return pixel_unshuffle(input, self.downscale_factor)

class Pixelshuffle(nn.Module):
    def __init__(self,upscale_factor):
        super(Pixelshuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self,input):
        return F.pixel_shuffle(input, self.upscale_factor)

def sobel_filter(x):
    x_in = x
    x_in = F.pad(x_in,(1,1,1,1),'reflect')
    # print(x_in)

    sobel1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobel2 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel3 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
    sobel4 = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]

    sobel1 = torch.FloatTensor(sobel1).unsqueeze(0).unsqueeze(0)
    sobel2 = torch.FloatTensor(sobel2).unsqueeze(0).unsqueeze(0)
    sobel3 = torch.FloatTensor(sobel3).unsqueeze(0).unsqueeze(0)
    sobel4 = torch.FloatTensor(sobel4).unsqueeze(0).unsqueeze(0)

    sobel1 = sobel1.cuda()
    sobel2 = sobel2.cuda()
    sobel3 = sobel3.cuda()
    sobel4 = sobel4.cuda()

    _xb_1 = F.conv2d(x_in[:, 0:1, :, :], sobel1)
    _xg_1 = F.conv2d(x_in[:, 1:2, :, :], sobel1)
    _xr_1 = F.conv2d(x_in[:, 2:3, :, :], sobel1)
    y_1 = torch.cat([_xb_1, _xg_1, _xr_1], dim=1)

    _xb_2 = F.conv2d(x_in[:, 0:1, :, :], sobel2)
    _xg_2 = F.conv2d(x_in[:, 1:2, :, :], sobel2)
    _xr_2 = F.conv2d(x_in[:, 2:3, :, :], sobel2)
    y_2 = torch.cat([_xb_2, _xg_2, _xr_2], dim=1)

    _xb_3 = F.conv2d(x_in[:, 0:1, :, :], sobel3)
    _xg_3 = F.conv2d(x_in[:, 1:2, :, :], sobel3)
    _xr_3 = F.conv2d(x_in[:, 2:3, :, :], sobel3)
    y_3 = torch.cat([_xb_3, _xg_3, _xr_3], dim=1)

    _xb_4 = F.conv2d(x_in[:, 0:1, :, :], sobel4)
    _xg_4 = F.conv2d(x_in[:, 1:2, :, :], sobel4)
    _xr_4 = F.conv2d(x_in[:, 2:3, :, :], sobel4)
    y_4 = torch.cat([_xb_4, _xg_4, _xr_4], dim=1)
    # print(y_4.shape)

    shape = torch.zeros_like(x).unsqueeze(-1)
    y = torch.repeat_interleave(shape, repeats=4, dim=-1)
    # print(y.shape)  torch.Size([1, 3, 5, 5, 4])

    y[:, :, :, :, 0] = y_1
    y[:, :, :, :, 1] = y_2
    y[:, :, :, :, 2] = y_3
    y[:, :, :, :, 3] = y_4

    #验证
    # y_trans = y.permute(0,2,3,1,4)
    # print(y_trans)
    return y

def sobel_edge(x):
    sobel = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
             [[0, -1, -2], [1, 0, -1], [2, 1, 0]]]
    # print(num_sobel)  4
    sobel = torch.FloatTensor(sobel).unsqueeze(1)
    # print(sobel.shape) torch.Size([4, 1, 3, 3])
    sobel_tf = torch.repeat_interleave(sobel,repeats=3,dim=0)  # (12 1 3 3)
    # print(sobel_tf.shape)
    pad_x = F.pad(x,(1,1,1,1),'reflect')
    # pad_x = pad_x.permute(0, 2, 3, 1)
    y = F.conv2d(pad_x, sobel_tf, groups=3)
    # print(y.shape)  (1,12,5,5)
    shape = torch.zeros_like(y)
    shape[:, 0, :, :] = y[:, 0, :, :]
    shape[:, 4, :, :] = y[:, 1, :, :]
    shape[:, 8, :, :] = y[:, 2, :, :]
    shape[:, 1, :, :] = y[:, 3, :, :]
    shape[:, 5, :, :] = y[:, 4, :, :]
    shape[:, 9, :, :] = y[:, 5, :, :]
    shape[:, 2, :, :] = y[:, 6, :, :]
    shape[:, 6, :, :] = y[:, 7, :, :]
    shape[:, 10, :, :] = y[:, 8, :, :]
    shape[:, 3, :, :] = y[:, 9, :, :]
    shape[:, 7, :, :] = y[:, 10, :, :]
    shape[:, 11, :, :] = y[:, 11, :, :]
    shape = shape.permute(0,2,3,1)
    #test
    out = torch.reshape(shape,shape=(1,5,5,3,4))
    y_trans = out.permute(0, 3, 1, 2, 4)
    # print(y_trans)
    return y_trans

def build_window(windows):
    kernel = np.zeros(( windows * windows,1, windows, windows), dtype=np.float32)
    for i in range(windows * windows):
        kernel[i, 0, int(i / windows), int(i % windows)] = 1
    kernel = Variable(torch.from_numpy(kernel),requires_grad=True)
    return kernel

class FilterLayer(nn.Module):
    def __init__(self,windows):
        super(FilterLayer, self).__init__()
        self.kernel = build_window(windows)
        self.zeropadding = nn.ZeroPad2d(windows//2)

    def forward(self, x, f,window):
        x = self.zeropadding(x)
        # print(x.shape) (8, 3, 74, 74)
        _xb = F.conv2d(x[:, 0:1, :, :], self.kernel.cuda())
        # print(_xb.shape)  (8, 121, 64, 64)
        _xg = F.conv2d(x[:, 1:2, :, :], self.kernel.cuda())
        _xr = F.conv2d(x[:, 2:3, :, :], self.kernel.cuda())
        yb = torch.sum(_xb * f, dim=1, keepdim=True)
        yg = torch.sum(_xg * f, dim=1, keepdim=True)
        yr = torch.sum(_xr * f, dim=1, keepdim=True)
        y = torch.cat([yb, yg, yr], dim=1)
        # print(y.shape) (8, 3, 64, 64)
        return y

class residual_block(nn.Module):
    def __init__(self, inchannels):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, out_channels=64, kernel_size=3, stride=1, padding=3//2)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=64,kernel_size=3, stride=1, padding=3//2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.conv4 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)

    def forward(self, x):
        t1 = F.relu(self.conv1(x))
        t1 = torch.cat((x, t1), 1)
        # print(t1.shape) (8,192,128,128)
        t2 = F.relu(self.conv2(t1))
        t2 = torch.cat((t1,t2),1)
        # print(t2.shape)  (8,256,128,128)
        t3 = F.relu(self.conv3(t2))
        t3 = torch.cat((t2,t3),1)
        # print(t3.shape)    (8,320,128,128)
        t4 = F.relu(self.conv4(t3))
        t4 = torch.cat((t3,t4),1)

        t5 = F.relu(self.conv5(t4))
        t5 = torch.cat((t4,t5),1)
        return t5

class pre_block(nn.Module):
    def __init__(self, inchannels):
        super(pre_block, self).__init__()
        self.conv1 = nn.Conv2d(448, 256, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(448, 128, 1,stride=1, padding=0, bias=True)
        self.residual_block = residual_block(inchannels)

    def forward(self, x):
        t = x
        # print(t.shape) (8,128,128,128)
        t = self.residual_block(t)
        # print(t.shape) (8,448,128,128)
        a = F.relu(self.conv1(t))
        a = F.relu(self.conv2(a))
        a = F.relu(self.conv3(a))
        # print(a.shape)  (8,1,128,128)
        a =torch.sign(a)
        _a = (1-a)
        t = self.conv4(t)
        _t = torch.multiply(t, a)
        # print(_t.shape)  (8,128,128,128)
        _x = torch.multiply(x, _a)
        # print(_x.shape)  (8,128,128,128)
        # t =torch.cat((_x,_t),1)
        t = torch.add( _x, _t )
        return t

class pos_block(nn.Module):
    def __init__(self,inchannels):
        super(pos_block,self).__init__()
        self.residual_block = residual_block(inchannels)
        self.conv = nn.Conv2d(448, 128, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        t = x
        t = self.residual_block(t)
        t = F.relu(self.conv(t))
        return t

class filter_block(nn.Module):
    def __init__(self, inchannels, window):
        super(filter_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=inchannels,out_channels=window**2,kernel_size=3,stride=1,padding=3//2,bias=False)
        self.Filterlayer = FilterLayer(window)

    def forward(self,x,f,window):
        f = self.conv(f)
        # print(f.shape)  (8,121,64,64)
        f = F.softmax(f, dim=1) #问题！！！
        y = self.Filterlayer(x, f, window)
        return y

#branch-3
class network1(nn.Module):
    def __init__(self, nFilter):
        super(network1, self).__init__()
        self.conv1 = nn.Conv2d(20, nFilter*2, 3, stride=1, padding=3//2)
        self.pre_block = pre_block(nFilter*2)
        self.zeropadding = nn.ZeroPad2d(1)
        self.conv_stride = nn.Conv2d(128, nFilter*2, 3, stride=2)
        self.conv2 = nn.Conv2d(128,12,3,stride=1,padding=3//2)
        self.pos_block = pos_block(128)
        self.conv3 = nn.Conv2d(128, nFilter*4, 1)
        self.conv4 = nn.Conv2d(131, nFilter*2, 1)

        self.filter_block11 = filter_block(64, 11)
        self.filter_block9 = filter_block(64, 9)
        self.filter_block7 = filter_block(32, 7)

        self.pixelshuffle = Pixelshuffle(2)
        self.pixelunshuffle = PixelUnshuffle(2)

    def forward(self, x):
        output_list = []
        _x = self.pixelunshuffle(x)
        t1 = F.relu(self.conv1(_x))     #8m*8m
        t1 = self.pre_block(t1)
        # print(t1.shape) (8,128,128,128)

        t2 = self.zeropadding(t1)
        t2 = F.relu(self.conv_stride(t2))       #4m*4m
        t2 = self.pre_block(t2)
        # print(t2.shape)  (8,128,64,64)

        t3 = self.zeropadding(t2)
        t3 = F.relu(self.conv_stride(t3))       #2m*2m
        t3 = self.pre_block(t3)
        t3 = self.pre_block(t3)
        # print(t3.shape)     (8,128,32,32)

        _t3 = self.conv2(t3)
        _t3 = self.pixelshuffle(_t3)
        # print(_t3.shape)   (8, 3, 64, 64)

        t3 = self.pos_block(t3)
        t3 = F.relu(self.conv3(t3))
        t3 = self.pixelshuffle(t3)
        # print(t3.shape)    (8,64,64,64)

        t3_out = self.filter_block11(_t3, t3, 11)
        # print(t3_out.shape) (8, 3, 64, 64)
        output_list.append(t3_out)

        t2 = torch.cat((t3_out, t2), dim=1)
        # print(t2.shape) (8, 131, 64, 64)
        t2 = F.relu(self.conv4(t2))
        t2 = self.pre_block(t2)
        # print(t2.shape)     (8, 128, 64, 64)

        _t2 = self.conv2(t2)
        _t2 = self.pixelshuffle(_t2)
        # print(_t2.shape)  (8, 3, 128, 128)

        t2 = self.pos_block(t2)
        t2 = F.relu(self.conv3(t2))
        t2 = self.pixelshuffle(t2)
        # print(t2.shape)     (8, 64, 128, 128)
        t2_out = self.filter_block9(_t2, t2, 9)
        # print(t2_out.shape)  (8, 3, 128, 128)
        output_list.append(t2_out)

        t1 = torch.cat((t1,t2_out), dim=1)
        t1 = F.relu(self.conv4(t1))
        t1 = self.pre_block(t1)
        # print(t1.shape)  (1, 128, 128, 128)

        _t1 = self.conv2(t1)
        _t1 = self.pixelshuffle(_t1)
        t1 = self.pixelshuffle(t1)
        # print(t1.shape)  (1, 32, 256, 256)
        y = self.filter_block7(_t1, t1, 7)
        output_list.append(y)

        return output_list

#branch-3 修改
class network_new1(nn.Module):
    def __init__(self, nFilter):
        super(network_new1, self).__init__()
        self.conv1 = nn.Conv2d(20, nFilter*2, 3, stride=1, padding=3//2)
        self.pre_block = pre_block(nFilter*2)
        self.zeropadding = nn.ZeroPad2d(1)
        self.conv_stride = nn.Conv2d(128, nFilter*2, 3, stride=2)
        self.conv2 = nn.Conv2d(128,12,3,stride=1,padding=3//2)
        self.pos_block = pos_block(128)
        self.conv3 = nn.Conv2d(128, nFilter*4, 1)
        self.conv4 = nn.Conv2d(131, nFilter*2, 1)

        self.filter_block11 = filter_block(64, 11)
        self.filter_block9 = filter_block(64, 9)
        self.filter_block7 = filter_block(64, 7)

        self.pixelshuffle = Pixelshuffle(2)
        self.pixelunshuffle = PixelUnshuffle(2)

    def forward(self, x):
        output_list = []
        _x = self.pixelunshuffle(x)
        t1 = F.relu(self.conv1(_x))     #8m*8m
        # print(t1.shape) (8,128,128,128)

        t2 = self.zeropadding(t1)
        t2 = F.relu(self.conv_stride(t2))       #4m*4m

        t3 = self.zeropadding(t2)
        t3 = F.relu(self.conv_stride(t3))       #2m*2m

        t3 = self.pre_block(t3)
        t3 = self.pre_block(t3)
        # print(t3.shape)     (8,128,32,32)

        _t3 = self.conv2(t3)
        _t3 = self.pixelshuffle(_t3)
        # print(_t3.shape)   (8, 3, 64, 64)

        t3 = self.pos_block(t3)
        t3 = F.relu(self.conv3(t3))
        t3 = self.pixelshuffle(t3)
        t3_out = self.filter_block11(_t3, t3, 11)
        # print(t3_out.shape) (8, 3, 64, 64)
        output_list.append(t3_out)

        t2 = self.pre_block(t2)
        t2 = torch.cat((t3_out, t2), dim=1)
        t2 = F.relu(self.conv4(t2))
        t2 = self.pre_block(t2)
        # print(t2.shape)     (8, 128, 64, 64)

        _t2 = self.conv2(t2)
        _t2 = self.pixelshuffle(_t2)
        # print(_t2.shape)  (8, 3, 128, 128)

        t2 = self.pos_block(t2)
        t2 = F.relu(self.conv3(t2))
        t2 = self.pixelshuffle(t2)
        t2_out = self.filter_block9(_t2, t2, 9)
        # print(t2_out.shape)  (8, 3, 128, 128)
        output_list.append(t2_out)

        t1 = self.pre_block(t1)
        t1 = torch.cat((t1,t2_out), dim=1)
        t1 = F.relu(self.conv4(t1))
        t1 = self.pre_block(t1)
        # print(t1.shape)  (1, 128, 128, 128)

        _t1 = self.conv2(t1)
        _t1 = self.pixelshuffle(_t1)
        # print(_t1.shape)  ([2, 3, 256, 256])

        t1 = self.pos_block(t1)
        t1 = F.relu(self.conv3(t1))
        t1 = self.pixelshuffle(t1)
        y = self.filter_block7(_t1, t1, 7)
        # print(y.shape)  ([2, 3, 256, 256])
        output_list.append(y)

        return output_list

#branch-4
class network2(nn.Module):
    def __init__(self, nFilter):
        super(network2, self).__init__()
        self.conv1 = nn.Conv2d(20, nFilter*2, 3, stride=1, padding=3//2)
        self.pre_block = pre_block(nFilter*2)
        self.zeropadding = nn.ZeroPad2d(1)
        self.conv_stride = nn.Conv2d(128, nFilter*2, 3, stride=2)
        self.conv2 = nn.Conv2d(128,12,3,stride=1,padding=3//2)
        self.pos_block = pos_block(128)
        self.conv3 = nn.Conv2d(128, nFilter*4, 1)
        self.conv4 = nn.Conv2d(131, nFilter*2, 1)

        self.filter_block13 = filter_block(64, 13)
        self.filter_block11 = filter_block(64, 11)
        self.filter_block9 = filter_block(64, 9)
        self.filter_block7 = filter_block(32, 7)

        self.pixelshuffle = Pixelshuffle(2)
        self.pixelunshuffle = PixelUnshuffle(2)

    def forward(self, x):
        output_list = []
        _x = self.pixelunshuffle(x)
        t1 = F.relu(self.conv1(_x))     #8m*8m
        t1 = self.pre_block(t1)
        # print(t1.shape) (8,128,128,128)

        t2 = self.zeropadding(t1)
        t2 = F.relu(self.conv_stride(t2))       #4m*4m
        t2 = self.pre_block(t2)
        # print(t2.shape)  (8,128,64,64)

        t3 = self.zeropadding(t2)
        t3 = F.relu(self.conv_stride(t3))       #2m*2m
        t3 = self.pre_block(t3)
        # print(t3.shape)     (8,128,32,32)

        t4 = self.zeropadding(t3)
        t4 = F.relu(self.conv_stride(t4))       #1m*1m
        t4 = self.pre_block(t4)
        t4 = self.pre_block(t4)
        # print(t4.shape)     (8,128,16,16)

        _t4 = self.conv2(t4)
        _t4 = self.pixelshuffle(_t4)
        # print(_t4.shape) (8,3,16,16)

        t4 = self.pos_block(t4)
        t4 = F.relu(self.conv3(t4))
        t4 = self.pixelshuffle(t4)
        # print(t4.shape)     (8,64,64,64)

        t4_out = self.filter_block13(_t4,t4,13)
        # print(t4_out.shape)  (8,3,16,16)
        output_list.append(t4_out)

        t3 = torch.cat((t4_out, t3), dim=1)
        t3 = F.relu(self.conv4(t3))
        t3 = self.pre_block(t3)
        # print(t3.shape)  (8,128,16,16)

        _t3 = self.conv2(t3)
        _t3 = self.pixelshuffle(_t3)
        # print(_t3.shape)   (8, 3, 64, 64)

        t3 = self.pos_block(t3)
        t3 = F.relu(self.conv3(t3))
        t3 = self.pixelshuffle(t3)
        # print(t3.shape)    (8,64,64,64)
        t3_out = self.filter_block11(_t3, t3, 11)
        # print(t3_out.shape) (8, 3, 64, 64)
        output_list.append(t3_out)

        t2 = torch.cat((t3_out, t2), dim=1)
        # print(t2.shape) (8, 131, 64, 64)
        t2 = F.relu(self.conv4(t2))
        t2 = self.pre_block(t2)
        # print(t2.shape)     (8, 128, 64, 64)

        _t2 = self.conv2(t2)
        _t2 = self.pixelshuffle(_t2)
        # print(_t2.shape)  (8, 3, 128, 128)

        t2 = self.pos_block(t2)
        t2 = F.relu(self.conv3(t2))
        t2 = self.pixelshuffle(t2)
        # print(t2.shape)     (8, 64, 128, 128)
        t2_out = self.filter_block9(_t2, t2, 9)
        # print(t2_out.shape)  (8, 3, 128, 128)
        output_list.append(t2_out)

        t1 = torch.cat((t1,t2_out), dim=1)
        t1 = F.relu(self.conv4(t1))
        t1 = self.pre_block(t1)
        # print(t1.shape)  (1, 128, 128, 128)

        _t1 = self.conv2(t1)
        _t1 = self.pixelshuffle(_t1)
        # print(_t1.shape)    (8, 3, 128, 128)

        t1 = self.pixelshuffle(t1)
        print(t1.shape) 
        y = self.filter_block7(_t1, t1, 7)
        output_list.append(y)

        return output_list

#branch-2
class network3(nn.Module):
    def __init__(self, nFilter):
        super(network3, self).__init__()
        self.conv1 = nn.Conv2d(20, nFilter*2, 3, stride=1, padding=3//2)
        self.pre_block = pre_block(nFilter*2)
        self.zeropadding = nn.ZeroPad2d(1)
        self.conv_stride = nn.Conv2d(128, nFilter*2, 3, stride=2)
        self.conv2 = nn.Conv2d(128,12,3,stride=1,padding=3//2)
        self.pos_block = pos_block(128)
        self.conv3 = nn.Conv2d(128, nFilter*4, 1)
        self.conv4 = nn.Conv2d(131, nFilter*2, 1)

        self.filter_block9 = filter_block(64, 9)
        self.filter_block7 = filter_block(32, 7)

        self.pixelshuffle = Pixelshuffle(2)
        self.pixelunshuffle = PixelUnshuffle(2)

    def forward(self, x):
        output_list = []
        _x = self.pixelunshuffle(x)
        t1 = F.relu(self.conv1(_x))     #8m*8m
        t1 = self.pre_block(t1)
        # print(t1.shape) (8,128,128,128)

        t2 = self.zeropadding(t1)
        t2 = F.relu(self.conv_stride(t2))       #4m*4m
        t2 = self.pre_block(t2)
        t2 = self.pre_block(t2)
        # print(t2.shape)  (8,128,64,64)

        _t2 = self.conv2(t2)
        _t2 = self.pixelshuffle(_t2)
        # print(_t2.shape)  (8, 3, 128, 128)

        t2 = self.pos_block(t2)
        t2 = F.relu(self.conv3(t2))
        t2 = self.pixelshuffle(t2)
        # print(t2.shape)     (8, 64, 128, 128)
        t2_out = self.filter_block9(_t2, t2, 9)
        # print(t2_out.shape)  (8, 3, 128, 128)
        output_list.append(t2_out)

        t1 = torch.cat((t1,t2_out), dim=1)
        t1 = F.relu(self.conv4(t1))
        t1 = self.pre_block(t1)
        # print(t1.shape)  (1, 128, 128, 128)

        _t1 = self.conv2(t1)
        _t1 = self.pixelshuffle(_t1)
        t1 = self.pixelshuffle(t1)
        # print(t1.shape)  (1, 32, 256, 256)
        y = self.filter_block7(_t1, t1, 7)
        output_list.append(y)

        return output_list

#branch-1
class network4(nn.Module):
    def __init__(self, nFilter):
        super(network4, self).__init__()
        self.conv1 = nn.Conv2d(20, nFilter*2, 3, stride=1, padding=3//2)
        self.pre_block = pre_block(nFilter*2)
        self.zeropadding = nn.ZeroPad2d(1)
        self.conv_stride = nn.Conv2d(128, nFilter*2, 3, stride=2)
        self.conv2 = nn.Conv2d(128,12,3,stride=1,padding=3//2)
        self.pos_block = pos_block(128)
        self.conv3 = nn.Conv2d(128, nFilter*4, 1)

        self.filter_block7 = filter_block(32, 7)

        self.pixelshuffle = Pixelshuffle(2)
        self.pixelunshuffle = PixelUnshuffle(2)

    def forward(self, x):
        output_list = []
        _x = self.pixelunshuffle(x)
        t1 = F.relu(self.conv1(_x))     #8m*8m
        t1 = self.pre_block(t1)
        t1 = self.pre_block(t1)
        # print(t1.shape)  (1, 128, 128, 128)

        _t1 = self.conv2(t1)
        _t1 = self.pixelshuffle(_t1)
        t1 = self.pixelshuffle(t1)
        # print(t1.shape)  (1, 32, 256, 256)
        y = self.filter_block7(_t1, t1, 7)
        output_list.append(y)

        return output_list




