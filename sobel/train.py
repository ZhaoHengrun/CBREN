from __future__ import print_function
import argparse, os
import torch
import random
import math
import wandb
import sys
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import get_training_set, get_test_set
import torch.optim as optim
from model import network1, sobel_filter


os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use the chosen gpu

def adjust_learning_rate(optimizer, epoch):
    global lr
    global adjust_lr_flag
    global stop_flag

    if adjust_lr_flag is True:
        lr = lr * 0.5
        if lr < 5e-6:
            stop_flag = True
        print('-------------adjust lr to [{:.7}]-------------'.format(lr))
        adjust_lr_flag = False
    return lr

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def train(epoch,model,optimizer, criterion):
    global save_flag
    global psnr_avr

    avr_loss = 0
    iteration_loss = 0

    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, target2, target3 = Variable(batch[0]), Variable(batch[1], requires_grad=False),\
                         Variable(batch[2], requires_grad=False),Variable(batch[3], requires_grad=False)
        input = torch.squeeze(input,dim=0)
        # print(input)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            target2 = target2.cuda()
            target3 = target3.cuda()

        _out = model(input)

        sobel_target = sobel_filter(target)
        sobel_target2 = sobel_filter(target2)
        sobel_target3 = sobel_filter(target3)

        _out_sobel0 = sobel_filter(_out[0])
        _out_sobel1 = sobel_filter(_out[1])
        _out_sobel2 = sobel_filter(_out[2])

        l1_loss_3 = criterion(_out[0], target3)
        S_loss_3 = criterion(_out_sobel0,sobel_target3)
        loss_32 = l1_loss_3 + 0.25*S_loss_3

        l1_loss_2 = criterion(_out[1], target2)
        S_loss_2 = criterion(_out_sobel1, sobel_target2)
        loss_64 = l1_loss_2 + 0.25 * S_loss_2

        l1_loss_1 = criterion(_out[2], target)
        S_loss_1 = criterion(_out_sobel2, sobel_target)
        loss_128 = l1_loss_1 + 0.25 * S_loss_1

        loss = loss_32 + loss_64 + loss_128
        avr_loss += loss.item()
        iteration_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write("===> Epoch[{}]({}/{}): Loss: {:.6f}\r".
                         format(epoch, iteration, len(training_data_loader), loss.item()))

    avr_loss = avr_loss / len(training_data_loader)
    print("===> Epoch [{}] Complete: Avg.Loss: {:.6f}".format(epoch, avr_loss))

    if opt.visualization == 'wandb':
        wandb.log({"Train avr_loss": avr_loss,'Valid PSNR': psnr_avr, 'Learning Rate': lr})

def test(epoch, model,criterion_val):
    global adjust_lr_flag
    global psnr_avr
    global last_psnr_avr_1
    global last_psnr_avr_2
    global last_psnr_avr_3
    global last_psnr_avr_4
    global last_psnr_avr_5
    global psnr_max
    global save_flag
    last_psnr_avr_5 = last_psnr_avr_4
    last_psnr_avr_4 = last_psnr_avr_3
    last_psnr_avr_3 = last_psnr_avr_2
    last_psnr_avr_2 = last_psnr_avr_1
    last_psnr_avr_1 = psnr_avr
    if (epoch % 1) == 0:
        psnr_sum = 0
        with torch.no_grad():
            for iteration, batch in enumerate(testing_data_loader,1):
                input,target,target2,target3 = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False), \
                                                Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)

                model.eval()
                if opt.cuda:
                    input = input.cuda()
                    target = target.cuda()

                _out = model(input)
                mse = criterion_val(_out[2], target)
                cacl_psnr = 10 * math.log10(1/mse.item())
                psnr_sum += cacl_psnr

        psnr_avr = psnr_sum /len(testing_data_loader)
        if psnr_max < psnr_avr:
            psnr_max = psnr_avr
            save_flag = True
            print('||||||||||||||||||||||best psnr is[{:.6f}]||||||||||||||||||||||'.format(psnr_max))
        if last_psnr_avr_1 < last_psnr_avr_5 and last_psnr_avr_2 < last_psnr_avr_5 and last_psnr_avr_3 < last_psnr_avr_5 \
                and last_psnr_avr_4 < last_psnr_avr_5 and psnr_avr < last_psnr_avr_5 and epoch > 5:
            adjust_lr_flag = True
            last_psnr_avr_1 = 0
            last_psnr_avr_2 = 0
            last_psnr_avr_3 = 0
            last_psnr_avr_4 = 0
            last_psnr_avr_5 = 0
        print('psnr_valid:[{:.6f}],last_psnr_avr_1:[{:.6f}],last_psnr_avr_2:[{:.6f}],last_psnr_avr_3:[{:.6f}],'
              'last_psnr_avr_4:[{:.6f}],last_psnr_avr_5:[{:.6f}]'
              .format(psnr_avr, last_psnr_avr_1, last_psnr_avr_2,
                      last_psnr_avr_3, last_psnr_avr_4, last_psnr_avr_5))

def save_checkpoint(epoch, model):
    global save_flag

    model_folder = "checkpoint/"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if (epoch % 5) == 0:
        torch.save(model, model_folder + "model_epoch_{}.pth".format(epoch))
        print("Checkpoint saved to {}".format(model_folder))
    if save_flag is True:
        torch.save(model, model_folder + "best_model.pth")
        save_flag = False
    torch.save(model, model_folder + "current_model.pth")

#Training settings
parser = argparse.ArgumentParser(description="PyTorch DenseNet")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--testbatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=50, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--visualization", default='wandb', type=str, help="wandb")
opt = parser.parse_args()
print(opt)

psnr_avr = 0
last_psnr_avr_1 = 0
last_psnr_avr_2 = 0
last_psnr_avr_3 = 0
last_psnr_avr_4 = 0
last_psnr_avr_5 = 0
psnr_max = 0
lr = opt.lr
adjust_lr_flag = False
stop_flag = False
save_flag = True


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

print("===> Loading datasets")
train_set = get_training_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
test_set = get_test_set()
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchSize)

print("===> Building model")
model = network1(64)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

criterion = nn.L1Loss()
criterion_val = nn.MSELoss()

print("===> Setting GPU")
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    criterion_val = criterion_val.cuda()

# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

# optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

#using wandb
if opt.visualization == 'wandb':
    wandb.init(project="DF_256")

print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=lr)

print("===> Training")
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    if stop_flag is True:
        print('finish')
        break
    train(epoch, model, optimizer, criterion)
    test(epoch, model, criterion_val)
    save_checkpoint(epoch, model)




