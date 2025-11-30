import os
from PIL import Image
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch

from PIL import ImageEnhance, ImageDraw, ImageFilter

""" 수정
def colorEnhance(image):
    bright_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    #color_intensity = random.randint(0, 10) / 10.0
    color_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    #sharp_intensity = random.randint(0, 30) / 10.0
    sharp_intensity = random.randint(7, 13) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
"""

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height, c = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height, c])
    return Image.fromarray(np.uint8(img))


def randomCrop(image, label):

    image_width = image.size[0]
    image_height = image.size[1]
    border = (int)(image_width * 0.15)

    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    #crop_win_height = crop_win_width
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.05 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

"""
selected1
                transforms.RandomAffine(
                    degrees=90, translate=(0.15, 0.15),
                    scale=(0.90, 1.15), shear=5),
"""

def CutPaste(img, gt, r1, r2, x0, y0, fixed_ratio = False):  # [0, 60] => percentage: [0, 0.2]
    v1 = random.uniform(r1, r2) * img.size[0]
    v2 = v1
    if fixed_ratio is False:
        v2 = random.uniform(r1, r2) * img.size[1]

    #print(v1, v2)

    w, h = img.size
    x0 = int(max(0, x0 - v1 / 2.))
    y0 = int(max(0, y0 - v2 / 2.))
    x1 = int(min(w, x0 + v1))
    y1 = int(min(h, y0 + v2))

    xy = (x0, y0, x1, y1)

    if (x1-x0) < 3 or (y1-y0) < 3:
        return img, gt

    x_ = int(np.random.uniform(w))
    y_ = int(np.random.uniform(h))

    img_ = img.crop(xy)
    img.paste(img_, (x_, y_)) #, x_+img_.size[0], y_+img_.size[1]))

    gt_ = gt.crop(xy)
    gt.paste(gt_, (x_, y_)) #, x_+img_.size[0], y_+img_.size[1]))

    return img, gt


def CutPasteImage(img, r1, r2, x0, y0, fixed_ratio = False):  # [0, 60] => percentage: [0, 0.2]
    v1 = random.uniform(r1, r2) * img.size[0]
    v2 = v1
    if fixed_ratio is False:
        v2 = random.uniform(r1, r2) * img.size[1]

    w, h = img.size
    x0 = int(max(0, x0 - v1 / 2.))
    y0 = int(max(0, y0 - v2 / 2.))
    x1 = int(min(w, x0 + v1))
    y1 = int(min(h, y0 + v2))

    xy = (x0, y0, x1, y1)

    if (x1-x0) < 3 or (y1-y0) < 3:
        return img

    x_ = int(np.random.uniform(w))
    y_ = int(np.random.uniform(h))

    img_ = img.crop(xy)
    img.paste(img_, (x_, y_)) #, x_+img_.size[0], y_+img_.size[1]))

    return img

def Cutout(img, r1, r2, x0, y0, fixed_ratio = False):  # [0, 60] => percentage: [0, 0.2]
    v1 = random.uniform(r1, r2) * img.size[0]
    v2 = v1
    if fixed_ratio is False:
        v2 = random.uniform(r1, r2) * img.size[1]

    return CutoutBlr(img, v1, v2, x0, y0)  # CutoutAbs


# img = _Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(512), np.random.uniform(512))

def _Cutout(img, r1, r2, x0, y0, fixed_ratio = False):  # [0, 60] => percentage: [0, 0.2]
    v1 = random.uniform(r1, r2) * img.size[0]
    v2 = v1
    if fixed_ratio is False:
        v2 = random.uniform(r1, r2) * img.size[1]

    return CutoutAbs(img, v1, v2, x0, y0)  # CutoutAbs

def CutoutAbs(img, v1, v2, x0, y0):
    w, h = img.size
    x0 = int(max(0, x0 - v1 / 2.))
    y0 = int(max(0, y0 - v2 / 2.))
    x1 = int(min(w, x0 + v1))
    y1 = int(min(h, y0 + v2))

    xy = (x0, y0, x1, y1)

    if (x1-x0) < 3 or (y1-y0) < 3:
        return img

    r = random.randint(0, 2)

    if r == 0:
        color = (255)
    elif r == 1:
        color = (145)
    else:
        color = (0)
    #img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)

    return img

"""
def CutoutBlr(img, v, b, x0, y0):
    w, h = img.size
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))

    xy = (x0, y0, x1, y1)

    img_ = img.copy()
    img_ = img_.crop(xy)
    gaussianBlur = ImageFilter.GaussianBlur(np.random.choice(range(1, 6)))
    img_ = img_.filter(gaussianBlur)
    img = np.array(img)
    img[y0:y1, x0:x1] = np.array(img_)
    img = Image.fromarray(img)

    return img
"""

def CutoutBlr(img, v1, v2, x0, y0):
    w, h = img.size
    x0 = int(max(0, x0 - v1 / 2.))
    y0 = int(max(0, y0 - v2 / 2.))
    x1 = int(min(w, x0 + v1))
    y1 = int(min(h, y0 + v2))

    xy = (x0, y0, x1, y1)

    if (x1-x0) < 3 or (y1-y0) < 3:
        return img

    #img_ = img.copy()
    img_ = img.crop(xy)
    gaussianBlur = ImageFilter.GaussianBlur(np.random.choice(range(1, 6)))
    img_ = img_.filter(gaussianBlur)
    img.paste(img_, (x0, y0))
    #img = np.array(img)
    #img[y0:y1, x0:x1] = np.array(img_)
    #img = Image.fromarray(img)

    return img

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                #transforms.RandomVerticalFlip(p=0.3),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.Resize((self.trainsize, self.trainsize)),
                #transforms.RandomAffine(
                #    degrees=0, translate=(0.15, 0.15),
                #    scale=(0.8, 1.2), shear=5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                #transforms.RandomVerticalFlip(p=0.3),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.Resize((self.trainsize, self.trainsize)),
                #transforms.RandomAffine(
                #    degrees=0, translate=(0.15, 0.15),
                #    scale=(0.8, 1.2), shear=5),
                transforms.ToTensor()])
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                #transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        self.g_blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 3.0))
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

        self.affine_transform = transforms.Compose([
                #transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1),
                    scale=(0.9, 1.15), shear=5)]
        )

    def __getitem__(self, index):

        img = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

      #새롭게 added
        #if random.random() > 0.80:
        #    image = colorEnhance(image)
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        # seed = np.random.randint(1024)
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        img = self.affine_transform(img)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        gt = self.affine_transform(gt)

        #if random.random() > 0.7:
        #   img = self.color_jitter(img)

        #if random.random() > 0.70:
        #    gt = randomPeper(gt)

        """
        rr = random.uniform(0, 1)

        if rr > 0.8:
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            #img = CutPasteImage(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #             np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.1, 0.2, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.05, 0.1,  np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #             np.random.uniform(self.trainsize))
        elif rr > 0.6:
            img, gt = CutPaste(img, gt, 0.1, 0.2, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.05, 0.1, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.1, 0.2,  np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
        elif rr > 0.4:
            img, gt = CutPaste(img, gt, 0.1, 0.2,  np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.05, 0.1, np.random.uniform(self.trainsize),  np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.1, 0.2, np.random.uniform(self.trainsize),  np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, 0.05, 0.1, np.random.uniform(self.trainsize),  np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            #img = CutPasteImage(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #            np.random.uniform(self.trainsize))
            #img = _Cutout(img, random.uniform(0.02, 0.05), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            #img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
            #             np.random.uniform(self.trainsize))

        """
        """
        elif rr > 0.4:
            img, gt = CutPaste(img, gt, random.uniform(0.1, 0.2), True, np.random.uniform(self.trainsize),
                               np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, random.uniform(0.1, 0.2), True, np.random.uniform(self.trainsize),
                              np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, random.uniform(0.1, 0.2), True, np.random.uniform(self.trainsize),
                              np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, random.uniform(0.1, 0.2), True, np.random.uniform(self.trainsize),
                              np.random.uniform(self.trainsize))
        """
        """
        elif rr > 0.4:
            img, gt = CutPaste(img, gt, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                               np.random.uniform(self.trainsize))
            img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                         np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                               np.random.uniform(self.trainsize))
            #img = _Cutout(img, random.uniform(0.02, 0.05), True, np.random.uniform(self.trainsize),
            #              np.random.uniform(self.trainsize))
            img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                         np.random.uniform(self.trainsize))
        """
        """
        elif rr > 0.2:
            img, gt = CutPaste(img, gt, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                               np.random.uniform(self.trainsize))
            img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                               np.random.uniform(self.trainsize))
            img = _Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
            img, gt = CutPaste(img, gt, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize),
                               np.random.uniform(self.trainsize))
            img = Cutout(img, random.uniform(0.05, 0.1), True, np.random.uniform(self.trainsize), np.random.uniform(self.trainsize))
        """

        #if random.random() > 0.7:
        #    img = self.g_blur(img)

        #if random.random() > 0.60:
        #    image, gt = randomCrop(img, gt)

        #seed = np.random.randint(2147483647)  # make a seed with numpy generator
        # seed = np.random.randint(1024)
        #random.seed(seed)  # apply this seed to img tranfsorms
        #torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            img = self.img_transform(img)

        #random.seed(seed)  # apply this seed to img tranfsorms
        #torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return img, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True,
               augmentation=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
