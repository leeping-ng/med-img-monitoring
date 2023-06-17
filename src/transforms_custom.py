from skimage.util import random_noise
import torch
from torchvision import transforms


class ContrastTransform:
    def __init__(self, contrast_factor):
        """
        How much to adjust the contrast. Can be any non-negative number.
        0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
        """
        self.contrast_factor = contrast_factor

    def __call__(self, tensor):
        tensor = transforms.functional.adjust_contrast(tensor, self.contrast_factor)
        return tensor


class SaltPepperNoiseTransform:
    def __init__(self, amount):
        """
        Proportion of image pixels to replace with noise on range [0, 1]. Default : 0.05
        """
        self.amount = amount

    def __call__(self, tensor):
        tensor = tensor[0, :, :]
        tensor = torch.tensor(
            random_noise(
                tensor, mode="s&p", salt_vs_pepper=0.5, clip=True, amount=self.amount
            )
        )
        tensor = tensor.expand(3, 224, 224)
        return tensor
