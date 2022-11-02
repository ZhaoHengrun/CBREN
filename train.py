import argparse
import os
import sys
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import visdom
import wandb
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from dataset import *
from model import *

# from edsr import EDSR

parser = argparse.ArgumentParser(description="CBREN_LD")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--dataset_target", default='datasets/train/ntire_gt/', type=str, help="dataset path")
parser.add_argument("--dataset_input", default='datasets/train/ntire_qp37/', type=str,
                    help="dataset path")
parser.add_argument("--dataset_valid_gt", default='datasets/valid/ntire_gt/', type=str,
                    help="dataset path")
parser.add_argument("--dataset_valid_input", default='datasets/valid/ntire_qp37/', type=str,
                    help="dataset path")
parser.add_argument("--checkpoints_path", default='checkpoints/CBREN_skip/', type=str, help="checkpoints path")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=2000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=50,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=200")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
parser.add_argument("--device", default='0', type=str, help="which gpu to use")
parser.add_argument("--visualization", default='wandb', type=str, help="none or wandb or visdom")
opt = parser.parse_args()

min_avr_loss = 99999999
epoch_avr_loss = 0
n_iter = 0
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

if opt.visualization == 'visdom':
    vis = visdom.Visdom(env='VECNN')


def main():
    global opt, model
    global stop_flag

    # torch.cuda.set_device(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device

    if opt.visualization == 'wandb':
        wandb.init(project="VECNN")

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = TrainDatasetSingleFrame(target_path=opt.dataset_target, input_path=opt.dataset_input)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                                      shuffle=True, num_workers=opt.threads)

    valid_set = ValidDatasetSingleFrame(target_path=opt.dataset_valid_gt, input_path=opt.dataset_valid_input)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=1,
                                   shuffle=False, num_workers=opt.threads)

    print("===> Building model")
    pyramid_cells = (3, 2, 1, 1, 1, 1)

    model = CBREN(n_channels=64, n_pyramids=1,
                  n_pyramid_cells=pyramid_cells, n_pyramid_channels=64)

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        # model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    # if opt.visualization == 'wandb':
    #     wandb.watch(model)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint.state_dict(), strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if stop_flag is True:
            print('finish')
            break
        train(training_data_loader, optimizer, model, criterion, epoch)
        valid(valid_data_loader, model, epoch)
        save_checkpoint(model, epoch)


def adjust_learning_rate(optimizer, epoch):
    global lr
    global adjust_lr_flag
    global stop_flag
    # lr = opt.lr * (0.5 ** (epoch // opt.step))
    if adjust_lr_flag is True:
        lr = lr * 0.5
        if lr < 1e-6:
            stop_flag = True
        print('-------------adjust lr to [{:.7}]-------------'.format(lr))
        with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
            f.write('------epoch[{}], adjust lr to [{:.7}]------\n'.format(epoch, lr))
        adjust_lr_flag = False
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter
    global psnr_avr

    avr_loss = 0

    lr = adjust_learning_rate(optimizer, epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target = batch[0], batch[1]  # b c h w
        # save_image_tensor(input[1, 2, :, :].unsqueeze(0), '{}{}.png'.format('results/input/', iteration))
        # save_image_tensor(target[1, :, :].unsqueeze(0), '{}{}.png'.format('results/target/', iteration))

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        out = model(input)
        # save_image_tensor(out[1, :, :].unsqueeze(0), '{}{}.png'.format('results/output/', iteration))
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        avr_loss += loss.item()
        # sys.stdout.write("===> Epoch[{}]({}/{}): Loss: {:.6f}\r".
        #                  format(epoch, iteration, len(training_data_loader), loss.item()))

        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                            loss.item()))
    avr_loss = avr_loss / len(training_data_loader)

    if opt.visualization == 'wandb':
        wandb.log({'Loss': avr_loss, 'Valid PSNR': psnr_avr, 'Learning Rate': lr})
    elif opt.visualization == 'visdom':
        vis.line(Y=np.array([avr_loss]), X=np.array([epoch]),
                 win='loss',
                 opts=dict(title='loss'),
                 update='append'
                 )

    epoch_avr_loss = avr_loss
    print('\nepoch_avr_loss[{:.6f}]'.format(epoch_avr_loss))
    with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
        f.write('epoch[{}], loss[{:.10f}], valid[{:.6f}], lr[{:.7f}]\n'.format(epoch, epoch_avr_loss, psnr_avr, lr))


def save_checkpoint(model, epoch):
    global min_avr_loss
    global save_flag

    model_folder = opt.checkpoints_path
    if (epoch % 5) == 0:
        torch.save(model, model_folder + "model_epoch_{}.pth".format(epoch))
        print("Checkpoint saved to {}".format(model_folder))
    if save_flag is True:
        torch.save(model, model_folder + "best_model.pth")
        save_flag = False
    torch.save(model, model_folder + "current_model.pth")


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def valid(valid_data_loader, model, epoch):
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
        sys.stdout.write('valid processing\r')
        print('valid processing')
        with torch.no_grad():
            for iteration, batch in enumerate(valid_data_loader, 1):
                input, target = batch[0], batch[1]
                model.eval()
                if opt.cuda:
                    model = model.cuda()
                    input = input.cuda()

                output = model(input)
                output = output.cpu()
                # save_image_tensor(output, '{}{}.png'.format('results/valid/', iteration))
                psnr_sum += calc_psnr(target, output).item()

        psnr_avr = psnr_sum / (len(listdir(opt.dataset_valid_gt)) * 20)
        if psnr_max < psnr_avr:
            psnr_max = psnr_avr
            save_flag = True
            print('||||||||||||||||||||||best psnr is[{:.6f}]||||||||||||||||||||||'.format(psnr_max))
            with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
                f.write('||||||epoch[{}], best psnr[{:.6f}]||||||\n'.format(epoch, psnr_max))
        if last_psnr_avr_1 < last_psnr_avr_5 and last_psnr_avr_2 < last_psnr_avr_5 and last_psnr_avr_3 < last_psnr_avr_5 \
                and last_psnr_avr_4 < last_psnr_avr_5 and psnr_avr < last_psnr_avr_5 and epoch > 1:
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
        # with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
        #     f.write('epoch[{}], psnr[{:.6f}]'.format(epoch, psnr_avr))


if __name__ == "__main__":
    main()
