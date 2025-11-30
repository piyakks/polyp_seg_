# copyright ziqi-jin
import torch
import torch.nn as nn

import copy
import torchvision
from torchvision import transforms
import torch.nn.functional as F

from typing import Any, Dict, List, Tuple

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

# Basic Convolution Block
class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class MLP(nn.Module):
    """
    Multilayer perception block
    :param
    channels: int
        number of input/output channels
    reduction_ratio: int, default=16
        channel reduction ratio
    """

    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio
        self.fc1 = nn.Linear(channels, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_ch, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.mlp = MLP(channels=gate_ch, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # global average pooling
        att1 = F.avg_pool2d(x, kernel_size=x.shape[2:], stride=x.shape[2:])
        att1 = self.mlp(att1)
        # max pooling
        att2 = F.max_pool2d(x, kernel_size=x.shape[2:], stride=x.shape[2:])
        att2 = self.mlp(att2)
        att = att1 + att2
        scale = F.sigmoid(att).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.spatial = BasicConv(in_ch=2, out_ch=1, kernel_size=7, stride=1, padding=3, relu=False)

    def forward(self, x):
        # max pooling
        att1 = torch.max(x, 1)[0].unsqueeze(1)
        att2 = torch.mean(x, 1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.spatial(att)
        scale = F.sigmoid(att).expand_as(x)

        return x * scale


# Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channelGate = ChannelGate(gate_ch=gate_channels, reduction_ratio=reduction_ratio)  # channel attention
        self.spatialGate = SpatialGate()

    def forward(self, x):
        out = self.channelGate(x)
        out = self.spatialGate(out)

        return out

def zero_conv(input, output, kernal):
    layer=nn.Conv2d(input, output, kernal, stride=1,padding=0, bias=True)
    for p in layer.parameters():
        p.detach().zero_()
    return layer

def normalize_image(image, mean=[0.485], std=[0.229]):
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize(image)

def zero_conv(input,output,kernal):
    layer=nn.Conv2d(input,output,kernal,stride=1,padding=0,bias=True)
    for p in layer.parameters():
        p.detach().zero_()
    return layer

def color_equalize(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image_copy = img.clone()
    for i in range(3):
            image_copy[:, i, :, :] = image_copy[:, i, :, :] * std[i] + mean[i]  

    image_copy = image_copy * 255
    image_copy = image_copy.clamp(0, 255)
    image_copy = image_copy.type(torch.uint8)
    e_image = torchvision.transforms.functional.equalize(image_copy)
    e_image = e_image.type(torch.float32)
    e_image = e_image / 255.0
    n_image = normalize_image(e_image)

    return n_image


class Twin_SAM(nn.Module):

    def __init__(self, sam_lora1, sam_lora2, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Twin_SAM, self).__init__()

        self.sam_model1 = sam_lora1.sam
        self.sam_twin_lora_encoder = sam_lora2.sam.image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()
        #self.multimask_output = multimask_output
        self.zero_conv_input = zero_conv(3, 3, 1)
        self.zero_conv_output = zero_conv(256, 256, 1)
        self.rfb = RFB(512, 256)        

        for n, value in self.sam_model1.prompt_encoder.named_parameters():
            value.required_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        x = self.sam_model1.image_encoder(input_images)

        control_img = color_equalize(input_images)
        control = self.zero_conv_input(torch.abs(control_img - input_images))
        control = control + input_images
        x2 = self.sam_twin_lora_encoder(control)
        x2 = self.zero_conv_output(x2)

        x = self.rfb(torch.cat((x, x2), 1))

        sparse_embeddings, dense_embeddings = self.sam_model1.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_model1.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_model1.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_model1.image_encoder.img_size, self.sam_model1.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_model1.image_encoder.img_size - h
        padw = self.sam_model1.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
  
class Twin_SAM2(nn.Module):

    def __init__(self, sam_lora1, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Twin_SAM2, self).__init__()

        self.sam_model1 = sam_lora1.sam
        self.sam_twin_lora_encoder = copy.deepcopy(sam_lora1.sam.image_encoder)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        self.rfb1 = RFB(256, 256)        
        self.rfb2 = RFB(256, 256)
        self.device = device

        for n, value in self.sam_model1.prompt_encoder.named_parameters():
            value.required_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        x1 = self.sam_model1.image_encoder(input_images)

        x2 = self.sam_twin_lora_encoder(input_images)

        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        #x = self.rfb(torch.cat((x, x2), 1))
        x = x1 + x2
        #x = self.rfb(x)

        sparse_embeddings, dense_embeddings = self.sam_model1.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_model1.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_model1.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_model1.image_encoder.img_size, self.sam_model1.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_model1.image_encoder.img_size - h
        padw = self.sam_model1.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    

  
class Adapted_Twin_SAM(nn.Module):

    def __init__(self, sam, sam_lora_image_encoder, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Adapted_Twin_SAM, self).__init__()

        self.sam = sam
        self.sam_lora_image_encoder = sam_lora_image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        self.rfb = RFB(512, 256)        
        self.device = device

        for n, value in self.sam.image_encoder.named_parameters():
            value.required_grad = False

        for n, value in self.sam.prompt_encoder.named_parameters():
            value.required_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        x = self.sam.image_encoder(input_images)

        #color_equalized = color_equalize(input_images)
        #control = self.zero_conv_input(color_equalized)
        #control = (control + input_images) / 2
        x2 = self.sam_lora_image_encoder(input_images)
        #x2 = self.zero_conv_output(x2)

        x = self.rfb(torch.cat((x, x2), 1))
        x = x + x2
        #x = self.rfb(x)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        #outputs = {
        #    'masks': masks,
        #    'iou_predictions': iou_predictions,
        #    'low_res_logits': low_res_masks
        #}
        #return outputs
        return masks, iou_predictions

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
from random import randint
  
class Adapted_Twin_SAM2(nn.Module):

    def __init__(self, sam_lora, sam_image_encoder, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Adapted_Twin_SAM2, self).__init__()

        self.sam_lora = sam_lora
        self.sam_image_encoder = sam_image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        #self.rfb = RFB(256, 256)
        self.cbam1 = CBAM(256)  
        self.cbam2 = CBAM(256)
        self.rfb_fused = RFB(512, 256)      
        self.device = device

        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False

        #for param in self.sam_lora.prompt_encoder.parameters():
        #    param.requires_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        x1 = self.sam_image_encoder(input_images)       
        #x1 = self.cbam1(x1)

        x2 = self.sam_lora.image_encoder(input_images)
        
        #x = self.rfb_fused(x1 * x2) #self.rfb_fused(torch.cat((x2, x1), 1))

        #x1 = self.cbam1(x1)
        #x = self.rfb(torch.cat((x2, x), 1))
        #x = x1 + x2
        #if r == 0:
        #x = self.rfb_fused(torch.cat((3*x2, x1), 1))
        x = 3*x2 + x1
        #else:
        #    x = self.rfb_fused(torch.cat((x1, x2), 1))
        #x = self.cbam2((x1 + 2*x2)/3)
        x = self.cbam1(x)
        #x = self.rfb(x)

        sparse_embeddings, dense_embeddings = self.sam_lora.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_lora.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_lora.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        #outputs = {
        #    'masks': masks,
        #    'iou_predictions': iou_predictions,
        #    'low_res_logits': low_res_masks
        #}
        #return outputs
        return masks, iou_predictions

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_lora.image_encoder.img_size, self.sam_lora.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_lora.image_encoder.img_size - h
        padw = self.sam_lora.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    


class Twin_SAM_Lora(nn.Module):

    def __init__(self, sam_lora, sam_lora_image_encoder, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Twin_SAM_Lora, self).__init__()

        self.sam_lora = sam_lora
        self.sam_lora_image_encoder = sam_lora_image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        self.rfb = RFB(512, 256)
        self.cbam = CBAM(256)
        #self.rfb2 = RFB(256, 256)        
        self.device = device

        #for param in self.sam_image_encoder.parameters():
        #    param.requires_grad = False

        for param in self.sam_lora.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        x1 = self.sam_lora.image_encoder(input_images)

        x2 = self.sam_lora_image_encoder(input_images)

        #x2 = self.rfb(x2)
        x = self.rfb(torch.cat((x1, x2), 1))
        x = self.cbam(x)
        #x = x1 + x2
        #x = self.rfb(x)

        sparse_embeddings, dense_embeddings = self.sam_lora.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_lora.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_lora.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        #outputs = {
        #    'masks': masks,
        #    'iou_predictions': iou_predictions,
        #    'low_res_logits': low_res_masks
        #}
        #return outputs
        return masks, iou_predictions

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_lora.image_encoder.img_size, self.sam_lora.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_lora.image_encoder.img_size - h
        padw = self.sam_lora.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    

class Twin_SAM_Combined(nn.Module):

    def __init__(self, sam_lora1, sam_lora2, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Twin_SAM_Combined, self).__init__()

        self.sam_model1 = sam_lora1.sam
        self.sam_model2 = sam_lora2.sam
        self.combined_mask_decoder = copy.deepcopy(self.sam_model1.mask_decoder)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()
        #self.multimask_output = multimask_output
        self.zero_conv_input = zero_conv(3, 3, 1)
        self.zero_conv_output = zero_conv(256, 256, 1)
        self.rfb = RFB(512, 256)        

        for n, value in self.sam_model1.prompt_encoder.named_parameters():
            value.required_grad = False

        for n, value in self.sam_model2.prompt_encoder.named_parameters():
            value.required_grad = False            

    def forward(self, batched_input, multimask_output, image_size):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):

        input_images = self.preprocess(batched_input)
        color_equalized = color_equalize(input_images)

        mask1 = self.sam_model1(input_images,multimask_output, image_size)
        mask2 = self.sam_model2(color_equalized,multimask_output, image_size)


        x1 = self.sam_model1.image_encoder(input_images)
        x2 = self.sam_model2.image_encoder(color_equalized)

        #control_img = color_equalize(input_images)
        #control = self.zero_conv_input(torch.abs(control_img - input_images))
        #control = control + input_images
        #x2 = self.sam_twin_lora_encoder(control)
        #x2 = self.zero_conv_output(x2)

        x = self.rfb(torch.cat((x1, x2), 1))

        sparse_embeddings, dense_embeddings = self.sam_model1.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.combined_mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_model1.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        mask_combined = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': mask_combined,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_model1.image_encoder.img_size, self.sam_model1.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_model1.image_encoder.img_size - h
        padw = self.sam_model1.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    
    
    
  
 
class Sibling_SAM(nn.Module):

    def __init__(self, sam_lora_l, sam_lora_b_image_encoder, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Sibling_SAM, self).__init__()

        self.sam_lora = sam_lora_l
        self.twin_sam_lora_encoder = sam_lora_b_image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        self.rfb = RFB(512, 256)        
        self.cbam = CBAM(256)
        self.device = device

        #for n, value in self.sam_model1.prompt_encoder.named_parameters():
        #    value.required_grad = False
        
        for param in self.sam_lora.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        x1 = self.sam_lora.image_encoder(input_images)

        x2 = self.twin_sam_lora_encoder(input_images)

        x = self.rfb(torch.cat((x1, x2), 1))

        x = self.cbam(x)

        sparse_embeddings, dense_embeddings = self.sam_lora.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_lora.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_lora.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        #outputs = {
        #    'masks': masks,
        #    'iou_predictions': iou_predictions,
        #    'low_res_logits': low_res_masks
        #}
        #return outputs
        return masks, iou_predictions        

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_lora.image_encoder.img_size, self.sam_lora.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_lora.image_encoder.img_size - h
        padw = self.sam_lora.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
class Adapted_Mamba_SAM(nn.Module):

    def __init__(self, sam_lora, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Adapted_Mamba_SAM, self).__init__()

        self.sam_lora = sam_lora
        # self.sam_image_encoder = sam_image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        #self.rfb = RFB(256, 256)
        # self.cbam1 = CBAM(256)  
        # self.cbam2 = CBAM(256)
        # self.rfb_fused = RFB(512, 256)      
        self.device = device

        # for param in self.sam_image_encoder.parameters():
        #     param.requires_grad = False

        #for param in self.sam_lora.prompt_encoder.parameters():
        #    param.requires_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        # x1 = self.sam_image_encoder(input_images)       

        x2 = self.sam_lora.image_encoder(input_images)
        # x = 3*x2 + x1
        x=x2
        # x = self.cbam1(x)
        

        sparse_embeddings, dense_embeddings = self.sam_lora.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_lora.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_lora.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks( 
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        #outputs = {
        #    'masks': masks,
        #    'iou_predictions': iou_predictions,
        #    'low_res_logits': low_res_masks
        #}
        #return outputs
        
        return masks, iou_predictions

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_lora.image_encoder.img_size, self.sam_lora.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_lora.image_encoder.img_size - h
        padw = self.sam_lora.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
class Adapted_Mamba_SAM_dual(nn.Module):

    def __init__(self, sam,sam_lora, device, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
        super(Adapted_Mamba_SAM_dual, self).__init__()
        self.sam=sam
        self.sam_lora = sam_lora
        # self.sam_image_encoder = sam_image_encoder
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        #self.multimask_output = multimask_output
        #self.zero_conv_input = zero_conv(3, 3, 1)
        #self.zero_conv_output = zero_conv(256, 256, 1)
        #self.rfb = RFB(256, 256)
        self.cbam1 = CBAM(256)  
        # self.cbam2 = CBAM(256)
        # self.rfb_fused = RFB(512, 256)      
        self.device = device

        # for param in self.sam_image_encoder.parameters():
        #     param.requires_grad = False

        #for param in self.sam_lora.prompt_encoder.parameters():
        #    param.requires_grad = False

    def forward(self, batched_input, multimask_output, image_size):
        #if isinstance(batched_input, list):
        #    outputs = self.forward_test(batched_input, multimask_output)
        #else:
        #    outputs = self.forward_train(batched_input, multimask_output, image_size)
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std = self.pixel_std.to(self.device)

        outputs = self.generate_output(batched_input, multimask_output, image_size)
        return outputs

    def generate_output(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)

        # x1 = self.sam_image_encoder(input_images)       
        x1 = self.sam.image_encoder(input_images)
        x2 = self.sam_lora.image_encoder(input_images)
        x = 3*x2 + x1
        x = self.cbam1(x)
        

        sparse_embeddings, dense_embeddings = self.sam_lora.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam_lora.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam_lora.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks( 
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        #outputs = {
        #    'masks': masks,
        #    'iou_predictions': iou_predictions,
        #    'low_res_logits': low_res_masks
        #}
        #return outputs
        
        return masks, iou_predictions

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.sam_lora.image_encoder.img_size, self.sam_lora.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_lora.image_encoder.img_size - h
        padw = self.sam_lora.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x