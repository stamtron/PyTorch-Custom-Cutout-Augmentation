import random
import torch
from torchvision.transforms import functional as F
from skimage.exposure import equalize_adapthist
import numpy as np
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


class ImageEqualize:    
    def __init__(self, clip_limit, difference):
        self._clip_limit = clip_limit
        self._difference = difference

    def equalize(self, img):
        img = np.array(img)
        eimg = equalize_adapthist(img, clip_limit=self._clip_limit)  

        if self._difference:
            img = img - eimg
            img = img - img.min()
            img = img / img.max()
        else:
            return eimg

    def __call__(self, image, target):
        return self.equalize(image), target
    
def combine_masks(masks, mask_threshold):
    """Combine the masks labeled with their sequence number."""
    masks = masks.numpy()
    masks = (masks >= mask_threshold).astype(int)
    all_masks = np.zeros_like(masks[0])
    for i, mask in enumerate(masks, 1):
        mask_indices = mask.astype(bool)  # Convert mask values to boolean to get the indices
        all_masks[mask_indices] = i  # 
        #all_masks[mask] = i 
        #all_masks[mask == True] = i
    return all_masks
    
def visualize(img, boxes, masks, enhance=None, model=None):
    masks = combine_masks(masks, mask_threshold=0.5)
    
    #fig, axs = plt.subplots(2, 2, figsize=(14,14))
    fig, axs = plt.subplots(1, 2, figsize=(14,14))
    # Set individual titles for each subplot
    titles = ["Original Image", "Masks and Boxes"]

    # Plot the original image
    axs[0].imshow(img)
    axs[0].set_title(titles[0])

    # Plot the masks and boxes
    for i in range(len(boxes)):
        box = boxes[i]
        w = box[2] - box[0]
        h = box[3] - box[1]
        rect = matplotlib.patches.Rectangle(
            (box[0], box[1]), w, h, linewidth=1, edgecolor="lime", facecolor="none"
        )
        axs[1].add_patch(rect)
    axs[1].imshow(label2rgb(masks, img, bg_label=0))
    axs[1].set_title(titles[1])

    plt.show()
    
def imshow(image):
    # image is a tensor of shape (3, H, W)
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    plt.imshow(img)
    plt.show()

    
def denormalize_image(image):
    # Define ImageNet statistics for mean and standard deviation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalized_image = image.copy()
    for i in range(3):
        denormalized_image[:, :, i] = (denormalized_image[:, :, i] * std[i]) + mean[i]
    return denormalized_image

def visualize_sample(ds, sample_index, enhance=None, model=None):
    img, targets = ds[sample_index]
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    visualize(img, targets["boxes"], targets["masks"], enhance, model)
    
def visualize_coutout_sample(img, targets, enhance=None, model=None):
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    visualize(img, targets["boxes"], targets["masks"], enhance, model)
    
def visualize_predicted_boxes(img, boxes, enhance=None, model=None):
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    #fig, axs = plt.subplots(2, 2, figsize=(14,14))
    fig, axs = plt.subplots(1, 2, figsize=(14,14))
    # Set individual titles for each subplot
    titles = ["Original Image", "Predicted Boxes"]

    # Plot the original image
    axs[0].imshow(img)
    axs[0].set_title(titles[0])

    # Plot the masks and boxes
    axs[1].imshow(img)
    for i in range(len(boxes)):
        box = boxes[i]
        w = box[2] - box[0]
        h = box[3] - box[1]
        rect = matplotlib.patches.Rectangle(
            (box[0], box[1]), w, h, linewidth=1, edgecolor="orangered", facecolor="none"
        )
        axs[1].add_patch(rect)
    #axs[1].imshow(label2rgb(masks, img, bg_label=0))
    axs[1].set_title(titles[1])

    plt.show()
    
def visualize_predicted_boxes_and_gt(img, gt_boxes, pred_boxes, enhance=None, model=None):
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    #fig, axs = plt.subplots(2, 2, figsize=(14,14))
    fig, axs = plt.subplots(1, 2, figsize=(14,14))
    # Set individual titles for each subplot
    titles = ["Original Image", "Predicted Vs Ground Truth Boxes"]

    # Plot the original image
    axs[0].imshow(img)
    axs[0].set_title(titles[0])

    # Plot the masks and boxes
    axs[1].imshow(img)
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        w = box[2] - box[0]
        h = box[3] - box[1]
        rect = matplotlib.patches.Rectangle(
            (box[0], box[1]), w, h, linewidth=1, edgecolor="orangered", facecolor="none"
        )
        axs[1].add_patch(rect)
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        w = box[2] - box[0]
        h = box[3] - box[1]
        rect = matplotlib.patches.Rectangle(
            (box[0], box[1]), w, h, linewidth=1, edgecolor="lime", facecolor="none"
        )
        axs[1].add_patch(rect)
    #axs[1].imshow(label2rgb(masks, img, bg_label=0))
    axs[1].set_title(titles[1])

    plt.show()
    
def visualize_predicted_masks(img, pred_masks, mask_threshold, enhance=None, model=None):
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    masks = combine_masks(pred_masks, mask_threshold=mask_threshold)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
    masks = masks.transpose((1, 2, 0))
    #masks = np.dstack([masks, masks, masks])
    
    fig, axs = plt.subplots(1, 2, figsize=(14,14))
    # Set individual titles for each subplot
    titles = ["Original Image", "Predicted Masks"]

    # Plot the original image
    axs[0].imshow(img)
    axs[0].set_title(titles[0])

    print(masks.shape)
    print(img.shape)
    # Plot the masks and boxes
    axs[1].imshow(label2rgb(masks[:,:,0], img, bg_label=0))
    axs[1].set_title(titles[1])

    plt.show()
    
def visualize_predicted_masks_and_boxes(img, pred_masks, mask_threshold, gt_boxes, pred_boxes, enhance=None, model=None):
    img = img.numpy().transpose((1, 2, 0))
    img = denormalize_image(img)
    masks = combine_masks(pred_masks, mask_threshold=mask_threshold)
    print(masks.shape)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
    masks = masks.transpose((1, 2, 0))
    #masks = np.dstack([masks, masks, masks])
    
    fig, axs = plt.subplots(1, 3, figsize=(16,14))
    # Set individual titles for each subplot
    titles = ["Original Image", "Predicted Masks", "Predicted Vs Ground Truth Boxes"]

    # Plot the original image
    axs[0].imshow(img)
    axs[0].set_title(titles[0])

    print(masks.shape)
    print(img.shape)
    # Plot the masks and boxes
    axs[1].imshow(label2rgb(masks[:,:,0], img, bg_label=0))
    axs[1].set_title(titles[1])
    
    # Plot the masks and boxes
    axs[2].imshow(img)
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        w = box[2] - box[0]
        h = box[3] - box[1]
        rect = matplotlib.patches.Rectangle(
            (box[0], box[1]), w, h, linewidth=1, edgecolor="orangered", facecolor="none"
        )
        axs[2].add_patch(rect)
    if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            box = gt_boxes[i]
            w = box[2] - box[0]
            h = box[3] - box[1]
            rect = matplotlib.patches.Rectangle(
                (box[0], box[1]), w, h, linewidth=1, edgecolor="lime", facecolor="none"
            )
            axs[2].add_patch(rect)
    #axs[1].imshow(label2rgb(masks, img, bg_label=0))
    axs[2].set_title(titles[2])
    
    plt.show()