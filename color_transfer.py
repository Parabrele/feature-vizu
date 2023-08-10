import torch

import itertools

import images

"""
Classic color transfer
"""

def color_transfer(image, style):
    image = images.rgb_to_lab(image)
    style = images.rgb_to_lab(style)

    image = images.stat_transfer(image, style)
    image = images.lab_to_rgb(image)

    return image

"""
Now, color transfer with k-clustering.

Find k color clusters in both images using k-means++ and assign each pixel to the closest cluster.
Then, sort clusters by number of pixels, and for each cluster in the image find the closest cluster in the style image.
    TODO : try to find different ways to do this (either from most common to less, or reverse, or find the pair that minimises distance
    or add a penalisation for distance in number of pixels and find the pair that minimizes the distance,
    or the pairs that minimise the weighted distance, weighted by the number of pixels in each cluster)
"""

def flatten_image(image):
    image = image.view(3, -1)
    image = image.transpose(0, 1)
    return image

def kmeans(image, k):
    image = flatten_image(image)
    
    # Initialize clusters

    k0 = torch.randint(0, image.shape[0], (1,)).item()
    k0 = image[k0]

    kstack = torch.zeros((k, 3))
    kstack[0] = k0

    for i in range(1, k):
        distances = torch.cdist(image, kstack[:i]).min(dim=1).values
        max_index = distances.argmax()
        kstack[i] = image[max_index]
    
    # While clusters are not stable, reassign pixels to clusters and update clusters
    while True:
        distances = torch.cdist(image, kstack)
        closest = distances.argmin(dim=1)

        new_kstack = torch.zeros_like(kstack)
        for i in range(k):
            new_kstack[i] = image[closest == i].mean(dim=0)
        
        if torch.all(new_kstack == kstack):
            break
        else:
            kstack = new_kstack
    
    return kstack

def find_weights(image, clusters):
    """
    Find the weight of each cluster in the image by counting the number of pixels in each cluster.
    """

    distances = torch.cdist(image, clusters)
    closest = distances.argmin(dim=1)

    weights = torch.zeros(clusters.shape[0])
    for i in range(clusters.shape[0]):
        weights[i] = (closest == i).sum()

    return weights

def pair_clusters(image, style, image_clusters, style_clusters):
    """
    Find the pairs of cluster that minimise the weighted distance between them.
    """

    #first, find the weight of each cluster
    image_weights = find_weights(image, image_clusters)
    style_weights = find_weights(style, style_clusters)

    #then, go through each possible arrangement of clusters and find the one that minimises the weighted distance
    #to do so, go through all permutations of the image cluster, and pair them with unpermuted style clusters.
    #This will go through all possible arrangements of clusters.

    distances = torch.cdist(image_clusters, style_clusters)

    k = image_clusters.shape[0]
    for permutation in itertools.permutations(range(k)):
        permutation = torch.tensor(permutation)

        permutation_distances = distances[torch.arange(k), permutation]
        weight_diff = (image_weights - style_weights[permutation]).abs()

        permutation_distances = permutation_distances * weight_diff
        permutation_distances = permutation_distances.sum()

        if 'min_distance' not in locals() or permutation_distances < min_distance:
            min_distance = permutation_distances
            best_permutation = permutation
    
    return best_permutation

def kmeans_color_transfer(image, style, k=3):
    #TODO : test clustering on rgb and on lab images
    #TODO : test moving the basic rgb vector to find "rotation" of the color space
    #       -> maybe instead of initializing the clusters with random pixels, initialize them with the basic rgb vectors

    # compute the distance between each pixel and each cluster
    # then apply the correction to each pixel based on the softmax distance to the clusters

    ###
    # Fing the clusters and distances
    ###

    b, c, h, w = image.shape
    
    image_clusters = kmeans(image, k)
    style_clusters = kmeans(style, k)

    image_distances = torch.cdist(flatten_image(image), image_clusters)
    style_distances = torch.cdist(flatten_image(style), style_clusters)
    image_distance_to_style = torch.cdist(flatten_image(image), style_clusters)
    
    image_softmax = torch.softmax(-image_distances, dim=1)
    style_softmax = torch.softmax(-image_distance_to_style, dim=1)

    ###
    # Find statistics about the clusters
    ###

    image_means, image_std = [], []
    style_means, style_std = [], []

    imlab = flatten_image(images.rgb_to_lab(image))
    stlab = flatten_image(images.rgb_to_lab(style))

    for i in range(k):
        image_indexes = (image_distances.argmin(dim=1) == i)
        style_indexes = (style_distances.argmin(dim=1) == i)

        image_means.append(imlab[image_indexes].mean(dim=0))
        image_std.append(imlab[image_indexes].std(dim=0))

        style_means.append(stlab[style_indexes].mean(dim=0))
        style_std.append(stlab[style_indexes].std(dim=0))

    image_means = torch.stack(image_means)
    image_std = torch.stack(image_std)

    style_means = torch.stack(style_means)
    style_std = torch.stack(style_std)

    ###
    # Transfer style according to softmax distance
    ###

    # multiply the means and std by the softmax and sum it for each pixel

    # i : image pixel
    # c : color cluster
    # C : color channel
    softmax_image_means = torch.einsum('ic, cC -> iC', image_softmax, image_means)
    softmax_image_std = torch.einsum('ic, cC -> iC', image_softmax, image_std)

    softmax_style_means = torch.einsum('ic, cC -> iC', style_softmax, style_means)
    softmax_style_std = torch.einsum('ic, cC -> iC', style_softmax, style_std)

    # then, apply the correction to the image
    imlab = imlab - softmax_image_means
    imlab = imlab * (softmax_style_std / softmax_image_std)
    imlab = imlab + softmax_style_means

    imlab = imlab.view(b, h, w, 3).permute(0, 3, 1, 2)

    return images.lab_to_rgb(imlab)