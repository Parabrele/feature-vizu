import torch
from torch import nn

from torchvision import transforms

import numpy as np
from PIL import Image as PILImage

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_projection = torch.tensor([[0.26, 0.09, 0.02],
                                           [0.27, 0.00, -0.05],
                                           [0.27, -0.09, 0.03]]).to(device)

color_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
color_std = [0.229, 0.224, 0.225]


def preprocess_image(img):
    #This function takes in a tensor with already the right shape
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    processed = transform(img)

    return processed

def preprocess_PIL(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    processed = transform(img)
    return processed.unsqueeze(0)


# Apply random transformations to the image
TFORM = transforms.Compose([
    transforms.Pad(12),
    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    transforms.RandomAffine(
        degrees=2, translate=(0.02, 0.02), scale=(1.05, 1.05), shear=2
    ),
])

RESIZE = lambda x: transforms.Resize(x)

def imshow(img, size = (6, 6)):
    if len(img.shape) == 4:
        img = img.squeeze(0)
    
    img = torch.clamp(img, 0, 1)

    img_reshaped = torch.permute(img, (1, 2, 0)).cpu()
    # make the plot the same size as the image
    plt.figure(figsize=size, frameon=False)
    plt.axis('off')

    plt.imshow(img_reshaped.detach())

    plt.xticks([])
    plt.yticks([])

    plt.show()

def save_image(img, path):
    if len(img.shape) == 4:
        img = img.squeeze(0)

    img_reshaped = torch.permute(img, (1, 2, 0))
    img_reshaped = img_reshaped.cpu().detach().numpy()

    img_reshaped = np.clip(img_reshaped, 0, 1)

    img_reshaped = PILImage.fromarray((img_reshaped * 255).astype(np.uint8))

    img_reshaped.save(path)


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