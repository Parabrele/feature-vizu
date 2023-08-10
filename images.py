import time
import os

import torch
from torch import nn

from torchvision import transforms

import numpy as np
from PIL import Image as PILImage

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_mean = torch.tensor([0.485, 0.456, 0.406])
color_std = torch.tensor([0.229, 0.224, 0.225])

rgb_to_xyz = torch.tensor([[0.5141, 0.3239, 0.1604],
                           [0.2651, 0.6702, 0.0641],
                           [0.0241, 0.1228, 0.8444]])

xyz_to_lms = torch.tensor([[0.3897, 0.6890, -0.0787],
                           [-0.2298, 1.1834, 0.0464],
                           [0.0000, 0.0000, 1.0000]])

rgb_to_lms = xyz_to_lms @ rgb_to_xyz

log_color = torch.log10

lms_to_lab = torch.tensor([[1/np.sqrt(3), 0, 0],
                           [0, 1/np.sqrt(6), 0],
                            [0, 0, 1/np.sqrt(2)]], dtype=torch.float32) @ \
                torch.tensor([[1, 1, 1],
                              [1, 1, -2],
                              [1, -1, 0]], dtype=torch.float32)

""" Artifacet of an old code. Should decorrelate, but I don't believe it. """
color_projection = torch.tensor([[0.26, 0.09, 0.02],
                                 [0.27, 0.00, -0.05],
                                 [0.27, -0.09, 0.03]])

BandW_intensity = torch.tensor([0.299, 0.587, 0.114])

def load_image(image_path, size=None):
    image = PILImage.open(image_path).convert('RGB')
    if size is not None:
        image = transforms.Resize(size)(image)
    
    image = np.array(image)/255
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    return image

def to_pil_image(tensor):
    if isinstance(tensor, torch.Tensor):
        image = tensor.clamp(0, 1)
        image = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image = (image*255).astype(np.uint8)
        image = PILImage.fromarray(image)
    else:
        image = tensor
    return image

def save_image_aux(image, path):
    image = to_pil_image(image)
    image.save(path, quality=95)

def save_image(image, path):
    if isinstance(image, list):
        for i, im in enumerate(image):
            timestr = time.strftime("%Y%m%d-%H%M%S") + '_' + str(i)
            save_image_aux(im, timestr + '_' + path)
    else:
        save_image_aux(image, path)

def imshow(image, title=None, size=1):
    if isinstance(image, list):
        size = int((np.sqrt(len(image)) * 6)) * size
        fig, ax = plt.subplots(1, len(image), figsize=(size, size), frameon=False)
        if title is not None:
            ax[0].set_title(title)

        for i, im in enumerate(image):
            im = to_pil_image(im)
            ax[i].imshow(im, cmap='gray' if im.mode == 'L' else None)
            ax[i].axis('off')
    else:
        image = to_pil_image(image)
        
        plt.figure(frameon=False)
        plt.axis('off')

        plt.imshow(image, cmap='gray' if image.mode == 'L' else None)

        if title is not None:
            plt.title(title)

    plt.show()

########
# Color functions
########


def rgb_to_lab(image):
    # image size is (1, 3, H, W)
    image = torch.einsum('Cc, bchw -> bChw', rgb_to_lms, image) #TODO : verify that it is not Cc
    image = log_color(image + 1e-12)
    image = torch.einsum('Cc, bchw -> bChw', lms_to_lab, image)
    return image

def lab_to_rgb(image):
    # image size is (1, 3, H, W)
    image = torch.einsum('Cc, bchw -> bChw', lms_to_lab.inverse(), image)
    image = torch.pow(10, image)
    image = torch.einsum('Cc, bchw -> bChw', rgb_to_lms.inverse(), image)
    return image

def BlackAndWhite(image):
    if image.shape[1] == 1:
        return image
    
    image = image.squeeze(0).permute(1, 2, 0)
    image = image @ BandW_intensity
    image = image.unsqueeze(0).unsqueeze(0)

    return image

def stat_transfer(image, style):
    style_mean = style.mean(dim=(2, 3), keepdim=True)
    style_std = style.std(dim=(2, 3), keepdim=True)

    im_mean = image.mean(dim=(2, 3), keepdim=True)
    im_std = image.std(dim=(2, 3), keepdim=True)

    image = (image - im_mean) / im_std
    image = image * style_std + style_mean

    return image

########
# Kernel functions
########

def gaussian_kernel(n=5, sigma=1):
    kernel = torch.zeros((1, 1, n, n))

    for i in range(n):
        for j in range(n):
            kernel[0, 0, i, j] = -((i-n//2)**2 + (j-n//2)**2)/(2*sigma**2)
    kernel = torch.exp(kernel)
    
    return kernel/kernel.sum()

def edge_kernel():
    kernel = gaussian_kernel(5)
    kernel[0, 0, 2, 2] = 0
    m = kernel.sum()
    kernel = -kernel
    kernel[0, 0, 2, 2] = m
    
    return kernel

def sharpen_kernel():
    kernel = edge_kernel()
    kernel[0, 0, 2, 2] = 1 + kernel[0, 0, 2, 2]
    
    return kernel

def distance_kernel(n = 5):
	kernel = torch.zeros((n, n))
	sqrtn = torch.sqrt(torch.tensor(n))
	for i in range(n):
		for j in range(n):
			kernel[i, j] = ((i - n//2)/sqrtn) ** 2 + ((j - n//2)/sqrtn) **2
	kernel = torch.sqrt(kernel).unsqueeze(0).unsqueeze(0)

	return kernel/kernel.sum()

def distance_inverse_kernel(n = 5, eps = 1):
    kernel = distance_kernel(n)
    
    kernel = 1/(kernel + eps) ** 2

    return kernel/kernel.sum()

########
# Convolution functions
########

def convolve(image, kernel):
    padded_image = transforms.Resize((image.shape[2] + kernel.shape[2] - 1, image.shape[3] + kernel.shape[3] - 1))(image)
    return nn.functional.conv2d(padded_image, kernel, padding=0)

def blur(image, size=5, sigma=1):
    kernel = gaussian_kernel(size, sigma)
    if image.shape[1] == 3:
        kernel = torch.cat((kernel, kernel, kernel), dim=1)
    return convolve(image, kernel)

def edge(image):
    image = blur(image, 7, 1.414)
    image = BlackAndWhite(image)
    image -= image.mean()
    kernel = edge_kernel()
    image = convolve(image, kernel)

    mean, std = image.mean(), image.std()
    image = (image - mean)/std
    image = image.abs()
    image = image.clamp(0, 1)**2

    return image

def sharpen(image):
    image = blur(image, 7, 1.414)
    kernel = sharpen_kernel()
    return convolve(image, kernel)

def distance_blur(image, n = 5, eps = 1):
    kernel = distance_inverse_kernel(n, eps)
    return convolve(image, kernel)

def smooth_edge_detector(image, max_distance = 21):
    image = blur(image, 7, 1.414)
    image = edge(image)
    image = blur(image, max_distance, sigma = max_distance/4)
    image /= image.max()

    return image

########
# MISCELLANEOUS
########

def bw_np_to_torch(image):
    image = torch.tensor(image).float()
    image = image.unsqueeze(0).unsqueeze(0)
    return image

def to_map(image, lambda_):
    image = image.clone()
    mean, std = image.mean(), image.std()
    image[image < mean + lambda_*std] = 0
    image[image > 0] = 1
    return image


########
# Preprocessing for CNN models
########

def preprocess_image(img):
    return transforms.Normalize(mean=color_mean, std=color_std)(img)

def preprocess_PIL(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    processed = transform(img)
    return processed.unsqueeze(0)

########
# Image perturbation for feature visualization
########

TFORM = transforms.Compose([
    transforms.Pad(12),
    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    transforms.RandomAffine(
        degrees=2, translate=(0.02, 0.02), scale=(1.05, 1.05), shear=2
    ),
])

RESIZE = lambda x: transforms.Resize(x)

########
# Image modules for feature visualization
########

class ExistingImage(nn.Module):
    def __init__(self, name, train=True, resize=None):
        super().__init__()
        im = PILImage.open(name)
        if resize is not None:
            im = transforms.Resize(resize)(im)
        im = np.array(im)[:, :, :3]/255
        im = torch.Tensor(im)
        im = im.unsqueeze(0)
        img_reshaped = torch.permute(im, (0, 3, 1, 2))
        
        if train:
            self.image = nn.Parameter(img_reshaped)
        else:
            self.image = img_reshaped

    def forward(self):
        return self.image


class PixelImage(nn.Module):
    def __init__(self, shape, std = 1.0):
        super().__init__()
        self.image = nn.Parameter(torch.randn(shape) * std)

    def forward(self):
        return self.image


class FourierImage(nn.Module):
    def __init__(self, shape, std = 1.0):
        super().__init__()
        b, c, h, w = shape
        freq = self.aux_fft(h, w)
        init_size = (2, b, c) + freq.shape

        init_val = torch.randn(init_size) * std
        spectrum = torch.complex(init_val[0], init_val[1])

        scale = 1.0 / torch.maximum(freq, torch.tensor(1.0 / max(h, w))) ** 1
        scale *= torch.sqrt(torch.tensor(w * h))

        scaled_spectrum = spectrum * scale

        self.shape = shape
        self.spectrum = nn.Parameter(scaled_spectrum)
    
    def aux_fft(self, h, w):
        fy = torch.fft.fftfreq(h)[:,None]
        if w%2 == 0:
            fx = torch.fft.fftfreq(w)[:w//2+1]
        else:
            fx = torch.fft.fftfreq(w)[:w//2+2]
        return (fx**2 + fy**2).sqrt()

    def forward(self):
        b, c, h, w = self.shape
        img = torch.fft.irfft2(self.spectrum)
        img = img[:, :, :h, :w] / 4.0
        return img


class ExistingFourierImage(nn.Module):
    def __init__(self, name, resize=None):
        super().__init__()
        im = PILImage.open(name)

        if resize is not None:
            im = transforms.Resize(resize)(im)
        
        im = np.array(im)[:, :, :3]/255
        im = torch.Tensor(im)
        im = im.unsqueeze(0)
        img_reshaped = torch.permute(im, (0, 3, 1, 2))
        
        spectrum = torch.fft.rfft2(img_reshaped)

        self.spectrum = nn.Parameter(spectrum)

    def forward(self):
        img = torch.fft.irfft2(self.spectrum)
        return img
    

def to_rgb(img, decorrelate = True, sigmoid = True):
    if decorrelate:
        img = torch.einsum('bchw,dc->bdhw', img, color_projection)
        if not sigmoid:
            img += color_mean[None,:,None,None]
    
    if sigmoid:
        img = torch.sigmoid(img)
    else :
        img = preprocess_image(img)
    
    return img

class Image(nn.Module):
    def __init__(self, w, h = None, std = 1.0, decorrelate = True, fft = True):
        super().__init__()
        h = h or w
        shape = (1, 3, h, w)

        imtype = FourierImage if fft else PixelImage
        self.img = imtype(shape, std = std)
        self.decorrelate = decorrelate

    def forward(self):
        img = self.img()
        img = to_rgb(img, decorrelate = self.decorrelate, sigmoid = True)
        return img