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


class GammaTransform:
    def __init__(self, gamma_factor):
        """
        gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
        """
        self.gamma_factor = gamma_factor

    def __call__(self, tensor):
        tensor = transforms.functional.adjust_gamma(tensor, self.gamma_factor)
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


class SpeckleNoiseTransform:
    def __init__(self, variance):
        """
        Variance of random distribution. Used in 'gaussian' and 'speckle'. Note: variance = (standard deviation) ** 2. Default : 0.01
        https://dsp.stackexchange.com/questions/38664/what-does-mean-and-variance-do-in-gaussian-noise
        """
        self.variance = variance

    def __call__(self, tensor):
        tensor = tensor[0, :, :]
        tensor = torch.tensor(
            random_noise(tensor, mode="speckle", clip=True, var=self.variance)
        )
        tensor = tensor.expand(3, 224, 224).float()
        return tensor


class BlurTransform:
    def __init__(self, kernel_size):
        """
        kernel_size (sequence of python:ints or int): Gaussian kernel size.
        Can be a sequence of integers like (kx, ky) or a single integer for square kernels.

        sigma (sequence of python:floats or float, optional): Gaussian kernel standard deviation.
        Can be a sequence of floats like (sigma_x, sigma_y) or a single float to define the same sigma in both X/Y directions.
        If None, then it is computed using kernel_size as sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8. Default, None.
        """
        self.kernel_size = kernel_size

    def __call__(self, tensor):
        tensor = transforms.functional.gaussian_blur(tensor, self.kernel_size)
        return tensor


class SharpenTransform:
    def __init__(self, sharpness_factor):
        """
        How much to adjust the sharpness. Can be any non-negative number.
        0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
        """
        self.sharpness_factor = sharpness_factor

    def __call__(self, tensor):
        tensor = transforms.functional.adjust_sharpness(tensor, self.sharpness_factor)
        return tensor
