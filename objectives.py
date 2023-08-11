"""
This file contains the objective functions for feature visualization.
It can be used to study
    one inner neuron,
    a channel of a layer,
    a whole layer,
    a list of such objectives
"""

import torch
import torch.nn as nn

from images import preprocess_image

class Objective(object):
    """
    This object is here to make all objectives have the same type and behavior.
    It allows a user to easily combine linearly several objectives.
    """

    def __init__(self, obj_fct, name=""):
        self.obj_fct = obj_fct
        self.name = name
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            obj_fct = lambda T: other + self.obj_fct(T)
            name = self.name
        elif isinstance(other, Objective):
            obj_fct = lambda T: self.obj_fct(T) + other.obj_fct(T)
            name = self.name + ", " + other.name
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
        
        return Objective(obj_fct, name=name)
    
    def __neg__(self):
        return -1 * self
    
    def __sub__(self, other):
        return self + (-other)
    
    @staticmethod
    def __sum__(objs):
        obj_fct = lambda T: sum(obj.obj_fct(T) for obj in objs)
        name = ", ".join(obj.name for obj in objs)
        return Objective(obj_fct, name=name)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            obj_fct = lambda T: other * self.obj_fct(T)
            name = self.name
        elif isinstance(other, Objective):
            obj_fct = lambda T: self.obj_fct(T) * other.obj_fct(T)
            name = self.name + " and " + other.name
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))
        
        return Objective(obj_fct, name=name)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __call__(self, model, param, transform=None):
        if transform is not None:
            input = transform(param())
        else:
            input = param()
        T = model(input)
        return self.obj_fct(T)


class SaveOutput:
    """Forward pytorch hook"""
    def __init__(self, detach=False):
        self.output = None
        self.detach = detach

    def __call__(self, module, module_in, module_out):
        self.output = module_out.detach() if self.detach else module_out
        
    def clear(self):
        self.output = None


def get_hook(model, layer_name, get_handles=False):
    i = 0
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        if name == layer_name:
            hook = SaveOutput(detach=get_handles)
            handle = layer.register_forward_hook(hook)

            if get_handles:
                return hook, handle
            return hook

def remove_handles(model: torch.nn.Module, handles) -> None:
    if isinstance(handles, list):
        for handle in handles:
            remove_handles(model, handle)
    else:
        handles.remove()


def channel(model, layer_name, channel):
    """
    This objective function returns the activation of a single channel.

    Registers a forward hook on the layer to access the activation.
    Returns the activation of the channel. TODO : positive or negative ?

    Args:
        model: pytorch model
        layer_name: name of the layer
        channel: number of the channel to visualise
    """

    hook = get_hook(model, layer_name)

    def obj_fct(model_output):
        mean = hook.output[:, channel].mean()
        hook.clear()
        return mean
    
    name = "channel {} of layer {}".format(channel, layer_name)
    
    return Objective(obj_fct, name=name)

def layer(model, layer_name):
    """
    This objective function returns the activation of a whole layer.

    Registers a forward hook on the layer to access the activation.
    Returns the activation of the layer.

    Args:
        model: pytorch model
        layer_name: name of the layer
    """

    hook = get_hook(model, layer_name)

    def obj_fct(model_output):
        mean = hook.output.mean()
        hook.clear()
        return mean
    
    name = "layer {}".format(layer_name)
    
    return Objective(obj_fct, name=name)

def classe(class_id):
    """
    This objective function returns the activation of a class.

    Registers a forward hook on the layer to access the activation.
    Returns the activation of the class.

    Args:
        model: pytorch model
        layer_name: name of the layer
        class_id: number of the class to visualise
    """

    def obj_fct(model_output):
        return model_output[:, class_id].mean()
    
    name = "class {}".format(class_id)

    return Objective(obj_fct, name=name)


#####
# From now on, we will define functions useful for the demonic blender.
# The goal is to extract the style of an image and infuse it into another while keeping the important features of the latter.
#####

def mean_L1(source, target):
    return (source - target).abs().mean()

def feature_max(tensor):
    b, c, h, w = tensor.size()
    flat_tensor = tensor.view(b * c, h * w)

    corr = torch.max(flat_tensor, dim=1)

    return corr.values

def feature_mean(tensor):
    b, c, h, w = tensor.size()
    flat_tensor = tensor.view(b * c, h * w)

    corr = torch.mean(flat_tensor, dim=1)

    return corr

def feature_std(tensor):
    b, c, h, w = tensor.size()
    flat_tensor = tensor.view(b * c, h * w)

    corr = torch.std(flat_tensor, dim=1)

    return corr

def corr_norm(tensor):
    b, c, h, w = tensor.size()
    flat_tensor = tensor.view(b * c, h * w)

    squared = flat_tensor ** 2
    norm = torch.sqrt(torch.sum(squared, dim=1)) # dim : b * c
    norm_colunm = norm.view(b * c, 1)
    norm_matrix = torch.mm(norm_colunm, norm_colunm.t())

    return norm_matrix

def feature_correlation_matrix(tensor, normalize=True, ccoef=False):
    """
    See stream_difference for more details.

    Note : wtf is ccoef ?! (also, lower the learning rate)
    """
    b, c, h, w = tensor.size()
    flat_tensor = tensor.view(b * c, h * w)

    if ccoef:
        flat_tensor = flat_tensor - torch.mean(flat_tensor, dim=1, keepdim=True)
    
    corr = torch.mm(flat_tensor, flat_tensor.t())

    if normalize:
        norm = corr_norm(tensor)
        corr /= norm + 1e-8
    else:
        corr /= (b * c * h * w)
    return corr

def get_activations(layer_hook, transform=None):
    activations = []
    for hook in layer_hook:
        activation = hook.output
        if transform is not None:
            activation = transform(activation)

        activations.append(activation)
    
    return activations

def get_stream(model, layer_names, difference_to, transform=None):
    layer_hh = [get_hook(model, layer_name, get_handles=True) for layer_name in layer_names]
    layer_hook = [hh[0] for hh in layer_hh]
    layer_handles = [hh[1] for hh in layer_hh]
    model(preprocess_image(difference_to))
    remove_handles(model, layer_handles)

    activations = get_activations(layer_hook, transform=transform)

    return activations

def fusion_activations(activations, mode):
    nb_layers = len(activations[0])
    stacked = [0]*nb_layers
    for layer in range(nb_layers):
        stacked[layer] = torch.stack([activation[layer] for activation in activations], dim=0)

        if mode == "mean":
            stacked[layer] = torch.mean(stacked[layer], dim=0)
        elif mode == "max":
            stacked[layer] = torch.max(stacked[layer], dim=0).values
        elif mode == "sum":
            stacked[layer] = torch.sum(stacked[layer], dim=0)
    
    return stacked

def stream_difference(model, layer_names, difference_to, activation_loss=mean_L1, transform=None):
    """
    Measures the activation difference between the stream of the "difference_to" image and the stream of the "blended" (difference_from) image, which is being optimized.
    
    This function will be called without transforms for the original image we want to modify,
    and with transforms = feature_corelation_matrix for the image we want to extract the style from.

    Intuition :
        While passing an image through a CNN, the activations represent different level of abstractions, different features of an image.
        This raw information can tell us where to find curves, collor gradients, etc. in the image.
        This is the important feature we want to preserve from the original image.

        Now, we want to extract the "style" of the image. This is a bit different from the raw features and needs a definition.
        We will define the style of an image as the correlation between the different features of the image.
        To compute it we simply make the matrix of the dot product of the features of the image with themselves.
        The idea is as follows : if two features happen to be correlated in the "style" image, we want them to be correlated in the "blended" image as well.
                                 For example, if in the style image, the curve detectors are sistematically triggered when there is a line of yellow surrounded by blue,
                                 we want the same to happen in the blended image.
                                 If circles always co occur with color gradient between red and green, we want circles in the blended image to behave the same way.
        
        This way, the position of the curves, triangles, ... in the original image are preserved
        while the color, gradient, sharpness, ... are replaced by the ones of the style image.

        EDIT : Now we don't really care about the first part, we only care about the features.
                As the correlation matrix completely disregards the position of the features, they will be preserved.
                See the notebook for more details.
    """

    # First, compute model(difference_to) as it is constant and we need to get the hook
    # While computing it we will save the activations of the layers we are interested in
    # in layer_hook_to, and then we will disconnect those hook to avoid overriding them
    # in future computations.

    if isinstance(difference_to, list):
        activations = []
        for img in difference_to:
            activations.append(get_stream(model, layer_names, img, transform=transform))
        activations = fusion_activations(activations, mode="mean")
    else:
        activations = get_stream(model, layer_names, difference_to, transform=transform)
    
    # Now that we have initialized everything, we can define the objective function as usual
    layer_hook_from = [get_hook(model, layer_name) for layer_name in layer_names]

    def obj_fct(model_output):
        loss = 0
        for i, hook_from in enumerate(layer_hook_from):
            activation_from = hook_from.output

            if transform is not None:
                activation_from = transform(activation_from)
            
            loss += activation_loss(activation_from, activations[i])
        
        return loss
    
    name = "stream difference"

    return Objective(obj_fct, name=name)

def get_truncated_model(model, layer_names):
    max_layer = max([int(layer_name.split("_")[1]) for layer_name in layer_names])
    i = 0
    truncated_model = nn.Sequential()
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            i+=1

        truncated_model.add_module(layer.__class__.__name__ + "_" + str(i), layer)
        
        if i >= max_layer:
            break

    return truncated_model