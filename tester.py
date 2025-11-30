import torch
import torchvision
from torch.autograd import Variable
import os
import argparse
from datetime import datetime

#from utils.dataloader_org import test_dataset, get_loader
from utils2.dataloader_new240602 import test_dataset, get_loader
#from utils.data_val import get_loader, test_dataset
from utils2.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

from importlib import import_module

from utils2.utils import DiceLoss

from torch.nn.modules.loss import BCEWithLogitsLoss

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int,
                        default=1, help='epoch number')

parser.add_argument('--module', type=str, default='mamba_my')

parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')  # 1e-4

parser.add_argument('--gpus', type=int,
                        default=1, help='gpu number')

parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

parser.add_argument('--batchsize', type=int,
                       
                        default=1, help='training batch size')

parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
					
parser.add_argument('--num_classes', type=int,
                    default=0, help='output channel of network') #원래는 8	

parser.add_argument('--vit_name', type=str,
                    default='vit_h', help='select one vit model')

parser.add_argument('--clip', type=float,
                        default=0.6, help='gradient clipping margin')

parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

parser.add_argument('--decay_epoch', type=int,
                        default=150, help='every n epochs decay learning rate')

parser.add_argument('--train_path', type=str,
                        default='/mnt/d/Polyp/TrainDataset2', help='path to train dataset')

parser.add_argument('--test_path', type=str,
                        default='/mnt/d/Polyp/TestDataset2/Kvasir', help='path to testing Kvasir dataset')

parser.add_argument('--test_path2', type=str,
                        default='/mnt/d/Polyp/TestDataset2/CVC-ClinicDB', help='path to testing Kvasir dataset')

parser.add_argument('--test_path3', type=str,
                        default='/mnt/d/Polyp/TestDataset2/CVC-300', help='path to testing Kvasir dataset')
parser.add_argument('--test_path4', type=str,
                        default='/mnt/d/Polyp/TestDataset2/CVC-ColonDB', help='path to testing Kvasir dataset')
parser.add_argument('--test_path5', type=str,
                        default='/mnt/d/Polyp/TestDataset2/ETIS-LaribPolypDB', help='path to testing Kvasir dataset')

parser.add_argument('--train_save', type=str,
                        default='Extended_SAM')

opt = parser.parse_args()
# from tqdm import tqdm
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    #weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=45, stride=1, padding=22) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()
from PIL import Image

def test(model, path, imagesize, multimask_output, device):
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####

    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, imagesize)
    b = 0.0
    save_dir = "./result_masks/{}".format(path[26:])  # 저장할 디렉토리 경로
    os.makedirs(save_dir, exist_ok=True)
    for i in range(test_loader.size):
        #image, gt, name, _ = test_loader.load_data()
        image, gt, name = test_loader.load_data() # dataloader사용할 경우

        w, h = gt.size
        max_wh = np.max([w, h])
        hp = int((max_wh - h) / 2)
        wp = int((max_wh - w) / 2)
        #print("h,w", h, w)

        gt = np.asarray(gt, np.float32)

        gt /= (gt.max() + 1e-8)
        image = image.to(device)

        #res, _ = model(image)
        #out = model(image, imagesize)
        res, _ = model(image, multimask_output, image_size=imagesize)

        #res = out['masks']        

        #print("gt.shape", gt.shape)
        #res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = F.interpolate(res, size=max_wh, mode='bilinear', align_corners=False)
        #print("res.shape", res.shape)

        #
        res = torchvision.transforms.functional.crop(res, hp, 0, h, w)
        #
        #print("res.shape", res.shape)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pil_img = Image.fromarray((res* 255).astype('uint8'))
        filename = os.path.splitext(name)[0] + ".png"
        save_path = os.path.join(save_dir, filename)
        pil_img.save(save_path)

        # # (5) 이미지 저장
        
        # input = res
        # target = np.array(gt)
        # N = gt.shape
        # smooth = 1
        # input_flat = np.reshape(input, (-1))
        # target_flat = np.reshape(target, (-1))

        # intersection = (input_flat * target_flat)

        # loss = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        # a = '{:.4f}'.format(loss)
        # a = float(a)
        # b = b + a

    return 0
def test_org(model, path, imagesize, multimask_output, device):
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####

    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, imagesize)
    b = 0.0
    for i in range(test_loader.size):
        #image, gt, name, _ = test_loader.load_data()
        image, gt, name = test_loader.load_data() # dataloader사용할 경우
        gt = np.asarray(gt, np.float32)

        gt /= (gt.max() + 1e-8)
        image = image.to(device)

        res, _ = model(image, multimask_output, image_size=imagesize)
        #out = model(image, imagesize)

        #res = out['masks']     
        #res = F.interpolate(res, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)        

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        #res = res.cpu().detach().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))

        intersection = (input_flat * target_flat)

        loss = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a = '{:.4f}'.format(loss)
        a = float(a)
        b = b + a

    return b / test_loader.size


def train(opt, model, epoch, device, multimask_output, total_step):
    model.eval()
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(1)
    
    # ---- multi-scale training ----
    #size_rates = [0.75, 1.25, 1]
    #save_epoch = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 149]
    size_rates = [1]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    print('start')
    meandice = test(model, opt.test_path, opt.trainsize, multimask_output, device)
    print('clear1')
    meandice2 = test(model, opt.test_path2, opt.trainsize, multimask_output, device)
    print('clear2')
    meandice3 = test(model, opt.test_path3, opt.trainsize, multimask_output, device)
    print('clear3')
    meandice4 = test(model, opt.test_path4, opt.trainsize, multimask_output, device)
    print('clear4')
    meandice5 = test(model, opt.test_path5, opt.trainsize, multimask_output, device)
    print('clear5')
            #    torch.save(model.state_dict(), save_path + 'Swin-CNN-FM384-model-best.pth')
    # print('[Results]: ', meandice, meandice2, meandice3,meandice4)


if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mamba_my import insert_mamba_lora


    # #model = CODSamColorEqualize('weights/sam_vit_b_01ec64.pth', True, True, False).to(device)
    sam1, img_embedding_size1 = sam_model_registry[opt.vit_name](image_size=opt.trainsize,
                                                                num_classes=opt.num_classes,
                                                                checkpoint=None, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    for param in sam1.parameters():
        param.requires_grad = False

    pkg2 = import_module(opt.module)
    net2 = pkg2.insert_mamba_lora(sam1)

                            
    #res = model.sam.image_encoder(torch.rand(size=(2, 3, opt.trainsize, opt.trainsize)))
    #res2, _ = model(torch.rand(size=(2, 3, opt.trainsize, opt.trainsize)), multimask_output = False, image_size=opt.trainsize)    

    if opt.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False


    from Extended_SAM import Adapted_Mamba_SAM

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    model = Adapted_Mamba_SAM(net2.sam, device)     
    model.load_state_dict(torch.load('./model_checkpoint/model_best12,0.813,0.819,.pth',map_location='cuda'),strict=True)
    # model.load_state_dict(torch.load('./outs/Extended_SAM/Extended_SAM270,0.921,0.941,0.909.pth'),strict=True)
    # model = torch.load('./outs/Extended_SAM_re_a/Extended_SAM_re5,0.925,0.944,0.920.pth')
    # torch.save(model.state_dict(),'model_checkpoint/model_second.pth')
    if (opt.gpus > 1):
        model = nn.DataParallel(model)
    
    model = model.to(device)  
    #from fp_network.models import Extended_Hybrid_MiT_UPP

    #model = Extended_Hybrid_MiT_UPP(encoder1_name='B4', encoder2_name='resnet34', encoder_weights="imagenet",
    #                                          in_channel=3, classes2=1).to(device)

    #x = torch.rand(1, 3, 224, 224)
    #x = x.to(device)

    #x = model(x)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(0, opt.epoch):
        #if epoch % 20 == 0:
        #    adjust_lr(optimizer, opt.lr, epoch, 0.1, opt.decay_epoch)
        train(opt,  model, epoch+1, device, multimask_output, total_step)
        #scheduler.step()
