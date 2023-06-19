from torchvision import transforms

from transforms_custom import (
    ContrastTransform,
    GammaTransform,
    SaltPepperNoiseTransform,
    SpeckleNoiseTransform,
    BlurSharpenTransform,
)


PREPROCESS_TF = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(256, antialias=True),
        # transforms.CenterCrop(224),
    ]
)

TRAIN_TF = transforms.Compose(
    [
        PREPROCESS_TF,
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomResizedCrop(224, (0.9, 1), antialias=True),
    ]
)


CONTRAST_INC_TF = {
    "Contrast Unchanged": transforms.Compose([PREPROCESS_TF, ContrastTransform(1.0)]),
    "Contrast Increase 10.0": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(10.0)]
    ),
}


CONTRAST_DEC_TF = {
    "Contrast Unchanged": transforms.Compose([PREPROCESS_TF, ContrastTransform(1.0)]),
    "Contrast Decrease 0.2": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.2)]
    ),
}

GAMMA_INC_TF = {
    "Gamma Unchanged": transforms.Compose([PREPROCESS_TF, GammaTransform(1.0)]),
    "Gamma Increase 2.0": transforms.Compose([PREPROCESS_TF, GammaTransform(2.0)]),
}


GAMMA_DEC_TF = {
    "Gamma Unchanged": transforms.Compose([PREPROCESS_TF, GammaTransform(1.0)]),
    "Gamma Decrease 0.2": transforms.Compose([PREPROCESS_TF, GammaTransform(0.2)]),
}

SALT_PEPPER_NOISE_TF = {
    "Salt Pepper Noise 0.0": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0)]
    ),
    "Salt Pepper Noise 0.5": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.5)]
    ),
}

SPECKLE_NOISE_TF = {
    "Speckle Noise 0.0": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.0)]
    ),
    "Speckle Noise 0.1": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.1)]
    ),
}

BLUR_TF = {
    "Blur Unchanged": transforms.Compose([PREPROCESS_TF, BlurSharpenTransform(1.0)]),
    "Blur 0.2": transforms.Compose([PREPROCESS_TF, BlurSharpenTransform(0.2)]),
}

SHARPEN_TF = {
    "Sharpen Unchanged": transforms.Compose([PREPROCESS_TF, BlurSharpenTransform(1.0)]),
    "Sharpen 20.0": transforms.Compose([PREPROCESS_TF, BlurSharpenTransform(20.0)]),
}

MAGNIFY_TF = {
    "Magnify Unchanged": transforms.Compose(
        [
            PREPROCESS_TF,
        ]
    ),
    "Magnify 10%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(246, antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 20%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(270, antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 30%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(292, antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
}
