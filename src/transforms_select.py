from torchvision import transforms

from transforms_custom import (
    ContrastTransform,
    GammaTransform,
    SaltPepperNoiseTransform,
    SpeckleNoiseTransform,
    BlurTransform,
    SharpenTransform,
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

BLUR_TF = {
    "Blur Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Blur 12.5%": transforms.Compose([PREPROCESS_TF, BlurTransform(13)]),
    "Blur 25%": transforms.Compose([PREPROCESS_TF, BlurTransform(27)]),
    "Blur 50%": transforms.Compose([PREPROCESS_TF, BlurTransform(53)]),
    "Blur 100%": transforms.Compose([PREPROCESS_TF, BlurTransform(107)]),
}

BLUR_TF_EDGE = {
    "Blur Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Blur AUC 0.69": transforms.Compose([PREPROCESS_TF, BlurTransform(107)]),
    "Blur AUC 0.67": transforms.Compose([PREPROCESS_TF, BlurTransform(113)]),
}

SHARPEN_TF = {
    "Sharpen Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Sharpen 12.5%": transforms.Compose([PREPROCESS_TF, SharpenTransform(18.75)]),
    "Sharpen 25%": transforms.Compose([PREPROCESS_TF, SharpenTransform(37.5)]),
    "Sharpen 50%": transforms.Compose([PREPROCESS_TF, SharpenTransform(75.0)]),
    "Sharpen 100%": transforms.Compose([PREPROCESS_TF, SharpenTransform(150.0)]),
}

SHARPEN_TF_EDGE = {
    "Sharpen Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Sharpen AUC 0.69": transforms.Compose([PREPROCESS_TF, SharpenTransform(150)]),
    "Sharpen AUC 0.66": transforms.Compose([PREPROCESS_TF, SharpenTransform(255)]),
}

SALT_PEPPER_NOISE_TF = {
    "Salt Pepper Noise Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Salt Pepper Noise 12.5%": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0081)]
    ),
    "Salt Pepper Noise 25%": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0163)]
    ),
    "Salt Pepper Noise 50%": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0325)]
    ),
    "Salt Pepper Noise 100%": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.065)]
    ),
}

SALT_PEPPER_NOISE_TF_EDGE = {
    "Salt Pepper Noise Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Salt Pepper Noise AUC 0.69": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.065)]
    ),
    "Salt Pepper Noise AUC 0.5": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.115)]
    ),
}

SPECKLE_NOISE_TF = {
    "Speckle Noise Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Speckle Noise 12.5%": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.0075)]
    ),
    "Speckle Noise 25%": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.015)]
    ),
    "Speckle Noise 50%": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.03)]
    ),
    "Speckle Noise 100%": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.06)]
    ),
}

SPECKLE_NOISE_TF_EDGE = {
    "Speckle Noise Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Speckle Noise AUC 0.69": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.06)]
    ),
    "Speckle Noise AUC 0.5": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(15.0)]
    ),
}

CONTRAST_INC_TF = {
    "Contrast Inc Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Inc 12.5%": transforms.Compose([PREPROCESS_TF, ContrastTransform(1.75)]),
    "Contrast Inc 25%": transforms.Compose([PREPROCESS_TF, ContrastTransform(2.5)]),
    "Contrast Inc 50%": transforms.Compose([PREPROCESS_TF, ContrastTransform(4.0)]),
    "Contrast Inc 100%": transforms.Compose([PREPROCESS_TF, ContrastTransform(7.0)]),
}

CONTRAST_INC_TF_EDGE = {
    "Contrast Inc Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Inc AUC 0.69": transforms.Compose([PREPROCESS_TF, ContrastTransform(7)]),
    "Contrast Inc AUC 0.63": transforms.Compose([PREPROCESS_TF, ContrastTransform(80)]),
}

CONTRAST_DEC_TF = {
    "Contrast Dec Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Dec 12.5%": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.886)]),
    "Contrast Dec 25%": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.771)]),
    "Contrast Dec 50%": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.543)]),
    "Contrast Dec 100%": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.085)]),
}

CONTRAST_DEC_TF_EDGE = {
    "Contrast Dec Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Dec AUC 0.69": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.085)]
    ),
    "Contrast Dec AUC 0.5": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.03)]
    ),
}

GAMMA_INC_TF = {
    "Gamma Inc Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Inc 12.5%": transforms.Compose([PREPROCESS_TF, GammaTransform(1.74)]),
    "Gamma Inc 25%": transforms.Compose([PREPROCESS_TF, GammaTransform(2.48)]),
    "Gamma Inc 50%": transforms.Compose([PREPROCESS_TF, GammaTransform(3.95)]),
    "Gamma Inc 100%": transforms.Compose([PREPROCESS_TF, GammaTransform(6.9)]),
}

GAMMA_INC_TF_EDGE = {
    "Gamma Inc Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Inc AUC 0.69": transforms.Compose([PREPROCESS_TF, GammaTransform(6.9)]),
    "Gamma Inc AUC 0.5": transforms.Compose([PREPROCESS_TF, GammaTransform(12)]),
}


GAMMA_DEC_TF = {
    "Gamma Dec Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Dec 12.5%": transforms.Compose([PREPROCESS_TF, GammaTransform(0.883)]),
    "Gamma Dec 25%": transforms.Compose([PREPROCESS_TF, GammaTransform(0.765)]),
    "Gamma Dec 50%": transforms.Compose([PREPROCESS_TF, GammaTransform(0.53)]),
    "Gamma Dec 100%": transforms.Compose([PREPROCESS_TF, GammaTransform(0.06)]),
}

GAMMA_DEC_TF_EDGE = {
    "Gamma Dec Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Dec AUC 0.69": transforms.Compose([PREPROCESS_TF, GammaTransform(0.06)]),
    "Gamma Dec AUC 0.53": transforms.Compose([PREPROCESS_TF, GammaTransform(0.01)]),
}


MAGNIFY_TF = {
    "Magnify Unchanged": transforms.Compose(
        [
            PREPROCESS_TF,
        ]
    ),
    "Magnify 12.5%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.1), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 25%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.2), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 50%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.4), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 100%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.8), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
}

MAGNIFY_TF_EDGE = {
    "Magnify Unchanged": transforms.Compose(
        [
            PREPROCESS_TF,
        ]
    ),
    "Magnify AUC 0.69": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.8), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.5": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.25), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
}
