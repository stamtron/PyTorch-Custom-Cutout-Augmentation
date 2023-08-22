import random
from typing import Any, Dict, Tuple, Union
from collections import namedtuple
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional
from warnings import warn

# import matplotlib.patches
import numpy as np
import torch


class CustomCutOut_TensorImages():
    """CoarseDropout of square regions in the image.

    Args:
        num_holes (int): Number of regions to zero out.
        min_cutout_size (int): Minimum size of the cutout patch.
        max_cutout_size (int): Maximum size of the cutout patch.
        bbox_removal_threshold (float): Threshold for removing bounding boxes covered by cutout.
        fill_value (list of float): Value for dropped pixels.
        always_apply (bool): Whether to always apply the augmentation.
        p (float): Probability of applying the augmentation.

    Image types:
        Pytorch Tensor Image
    """
    def __init__(
        self,
        num_holes=12,
        min_cutout_size=15,
        max_cutout_size=65,
        bbox_removal_threshold=0.30,
        fill_value=[102/255, 205/255, 0], #torch.Tensor([0, 1, 0]),
        always_apply=False,
        p=0.5,
    ):
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size
        self.num_holes = num_holes
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.always_apply = always_apply
        self.p = p
        
        
    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple

        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            torch.randint(0, img_width - cutout_size + 1, (1,)).item(),
            torch.randint(0, img_height - cutout_size + 1, (1,)).item()
        )

    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout patch with given fill value and determines the position in the original image

        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = torch.randint(self.min_cutout_size, self.max_cutout_size + 1, (1,)).item()
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        #cutout_patch = torch.full((cutout_size, cutout_size, 3), self.fill_value)
        #return cutout_patch, cutout_size, cutout_position
        return cutout_size, cutout_position
    
    def _generate_random_variation(self):
        """
        Generates a random color variation based on the fill value.
        Returns:
            list: A list of three values representing RGB color channels.
        """
        # Initialize the variation list with the original fill value
        variation = [self.fill_value[0], self.fill_value[1], self.fill_value[2]]

        # Generate random variations for each RGB channel
        for i in range(3):
            # Add a random value in the range [-0.2, 0.2] to the channel
            variation[i] += random.uniform(-0.2, 0.2)

            # Clamp the value between 0 and 1 to ensure valid color values
            variation[i] = max(0, min(1, variation[i]))
        return variation
    
    def apply(self, image, targets, **params):
        """
        Applies the cutout augmentation on the given image

        :param image: The image tensor to be augmented
        :returns augmented image tensor
        """
        if self.always_apply or random.random() < self.p:
            image = image.clone()  # Don't change the original image
            #print(image.shape)
            for _ in range(self.num_holes):
                _, self.img_height, self.img_width = image.shape
                #cutout_patch, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
                cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
                    
                # Set instance variables to use later
                self.image = image
                self.cutout_pos = cutout_pos
                self.cutout_size = cutout_size

                #transposed_cutout_arr = cutout_patch.permute(2, 0, 1)
                
                # ImageNet mean and standard deviation values for normalization
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                
                # Choose fill value randomly
                # Generate 10 variations
                variations = [self._generate_random_variation() for _ in range(10)]

                # Randomly choose one variation
                random_variation = random.choice(variations)

                # Green color value in range [0, 1] normalized with ImageNet statistics
                green_color_normalized = torch.tensor(random_variation, dtype=torch.float32)
                #green_color_normalized = torch.tensor(self.fill_value, dtype=torch.float32)
                
                self.green_color_normalized = (green_color_normalized - mean) / std
                self.green_color_normalized = self.green_color_normalized.cpu() #cuda()
                image[:, cutout_pos.y:cutout_pos.y + cutout_size, cutout_pos.x:cutout_size + cutout_pos.x] = self.green_color_normalized[:, None, None]#transposed_cutout_arr
            
            filtered_boxes = []
            filtered_masks = []
            filtered_area = []
            filtered_labels = []
            filtered_iscrowd = []
            if len(targets['boxes']) > 0:
                for i in range(len(targets['boxes'])):
                    box, _ = self.apply_to_bbox_mask(targets['boxes'][i], targets['masks'][i])
                    if box is not None:
                        filtered_boxes.append(targets['boxes'][i])
                        filtered_masks.append(targets['masks'][i])
                        filtered_area.append(targets['area'][i])
                        filtered_labels.append(targets['labels'][i])
                        filtered_iscrowd.append(targets['iscrowd'][i])
                
                if len(filtered_boxes) > 0:        
                    filtered_boxes_tensor = torch.stack(filtered_boxes)
                    filtered_masks_tensor = torch.stack(filtered_masks)
                    filtered_area_tensor = torch.stack(filtered_area)
                    filtered_labels_tensor = torch.stack(filtered_labels)
                    filtered_iscrowd_tensor = torch.stack(filtered_iscrowd)
                    targets['boxes'] = filtered_boxes_tensor
                    targets['masks'] = filtered_masks_tensor
                    targets['area'] = filtered_area_tensor
                    targets['labels'] = filtered_labels_tensor
                    targets['iscrowd'] = filtered_iscrowd_tensor
        return image, targets
    
    def apply_to_bbox_mask(self, bbox, mask, **params):
        """
        Removes the bounding boxes and masks which are covered by the applied cutout

        :param bbox: A single bounding box coordinates in pascal_voc format
        :param mask: A single mask corresponding to the bounding box
        :returns transformed bbox's coordinates and the mask
        """

        x_min, y_min, x_max, y_max = tuple(map(int, bbox))

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = torch.sum(
            #(self.image[:, y_min:y_max, x_min:x_max] == self.fill_value).all(dim=0)
            (self.image[:, y_min:y_max, x_min:x_max] == self.green_color_normalized.unsqueeze(1).unsqueeze(2)).all(dim=0)
        )

        # Remove the bbox if it has more than some threshold of content inside the cutout patch
        if overlapping_size / bbox_size > self.bbox_removal_threshold:
            bbox = None
            mask = None

        return bbox, mask
    
    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')
    
