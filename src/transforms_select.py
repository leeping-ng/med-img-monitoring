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
    "Blur AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Blur AUC 0.82": transforms.Compose([PREPROCESS_TF, BlurTransform(23)]),
    "Blur AUC 0.78": transforms.Compose([PREPROCESS_TF, BlurTransform(37)]),
    "Blur AUC 0.74": transforms.Compose([PREPROCESS_TF, BlurTransform(53)]),
    "Blur AUC 0.70": transforms.Compose([PREPROCESS_TF, BlurTransform(97)]),
}

BLUR_TF_EDGE = {
    "Blur Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Blur AUC 0.69": transforms.Compose([PREPROCESS_TF, BlurTransform(107)]),
    "Blur AUC 0.67": transforms.Compose([PREPROCESS_TF, BlurTransform(113)]),
}

SHARPEN_TF = {
    "Sharpen AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Sharpen AUC 0.82": transforms.Compose([PREPROCESS_TF, SharpenTransform(24)]),
    "Sharpen AUC 0.78": transforms.Compose([PREPROCESS_TF, SharpenTransform(38)]),
    "Sharpen AUC 0.74": transforms.Compose([PREPROCESS_TF, SharpenTransform(46)]),
    "Sharpen AUC 0.70": transforms.Compose([PREPROCESS_TF, SharpenTransform(58)]),
    "Sharpen AUC 0.66": transforms.Compose([PREPROCESS_TF, SharpenTransform(80)]),
}

SHARPEN_TF_EDGE = {
    "Sharpen Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Sharpen AUC 0.69": transforms.Compose([PREPROCESS_TF, SharpenTransform(150)]),
    "Sharpen AUC 0.66": transforms.Compose([PREPROCESS_TF, SharpenTransform(255)]),
}

SALT_PEPPER_NOISE_TF = {
    "Salt Pepper Noise AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Salt Pepper Noise AUC 0.82": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.003)]
    ),
    "Salt Pepper Noise AUC 0.78": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.007)]
    ),
    "Salt Pepper Noise AUC 0.74": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.012)]
    ),
    "Salt Pepper Noise AUC 0.70": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.016)]
    ),
    "Salt Pepper Noise AUC 0.66": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.025)]
    ),
    "Salt Pepper Noise AUC 0.62": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.056)]
    ),
    "Salt Pepper Noise AUC 0.58": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.066)]
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
    "Speckle Noise AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Speckle Noise AUC 0.82": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.004)]
    ),
    "Speckle Noise AUC 0.78": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.01)]
    ),
    "Speckle Noise AUC 0.74": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.02)]
    ),
    "Speckle Noise AUC 0.70": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.035)]
    ),
    "Speckle Noise AUC 0.66": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.065)]
    ),
    "Speckle Noise AUC 0.62": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(2)]
    ),
    "Speckle Noise AUC 0.58": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(4)]
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
    "Contrast Inc AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Contrast Inc AUC 0.82": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(2.8)]
    ),
    "Contrast Inc AUC 0.78": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(4.0)]
    ),
    "Contrast Inc AUC 0.74": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(4.9)]
    ),
    "Contrast Inc AUC 0.70": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(6.0)]
    ),
    "Contrast Inc AUC 0.66": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(8.5)]
    ),
    "Contrast Inc AUC 0.62": transforms.Compose([PREPROCESS_TF, ContrastTransform(22)]),
    "Contrast Inc AUC 0.58": transforms.Compose([PREPROCESS_TF, ContrastTransform(50)]),
}

CONTRAST_INC_TF_EDGE = {
    "Contrast Inc Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Inc AUC 0.69": transforms.Compose([PREPROCESS_TF, ContrastTransform(7)]),
    "Contrast Inc AUC 0.63": transforms.Compose([PREPROCESS_TF, ContrastTransform(80)]),
}

CONTRAST_DEC_TF = {
    "Contrast Dec AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Contrast Dec AUC 0.82": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.21)]
    ),
    "Contrast Dec AUC 0.78": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.155)]
    ),
    "Contrast Dec AUC 0.74": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.135)]
    ),
    "Contrast Dec AUC 0.70": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.115)]
    ),
    "Contrast Dec AUC 0.66": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.10)]
    ),
    "Contrast Dec AUC 0.62": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.09)]
    ),
    "Contrast Dec AUC 0.58": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.075)]
    ),
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
    "Gamma Inc AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Gamma Inc AUC 0.82": transforms.Compose([PREPROCESS_TF, GammaTransform(2.9)]),
    "Gamma Inc AUC 0.78": transforms.Compose([PREPROCESS_TF, GammaTransform(4.0)]),
    "Gamma Inc AUC 0.74": transforms.Compose([PREPROCESS_TF, GammaTransform(5.2)]),
    "Gamma Inc AUC 0.70": transforms.Compose([PREPROCESS_TF, GammaTransform(6.4)]),
    "Gamma Inc AUC 0.66": transforms.Compose([PREPROCESS_TF, GammaTransform(7.4)]),
    "Gamma Inc AUC 0.62": transforms.Compose([PREPROCESS_TF, GammaTransform(8.2)]),
    "Gamma Inc AUC 0.58": transforms.Compose([PREPROCESS_TF, GammaTransform(8.8)]),
}

GAMMA_INC_TF_EDGE = {
    "Gamma Inc Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Inc AUC 0.69": transforms.Compose([PREPROCESS_TF, GammaTransform(6.9)]),
    "Gamma Inc AUC 0.5": transforms.Compose([PREPROCESS_TF, GammaTransform(12)]),
}


GAMMA_DEC_TF = {
    "Gamma Dec AUC 0.86": transforms.Compose([PREPROCESS_TF]),
    "Gamma Dec AUC 0.82": transforms.Compose([PREPROCESS_TF, GammaTransform(0.26)]),
    "Gamma Dec AUC 0.78": transforms.Compose([PREPROCESS_TF, GammaTransform(0.21)]),
    "Gamma Dec AUC 0.74": transforms.Compose([PREPROCESS_TF, GammaTransform(0.185)]),
    "Gamma Dec AUC 0.70": transforms.Compose([PREPROCESS_TF, GammaTransform(0.16)]),
    "Gamma Dec AUC 0.66": transforms.Compose([PREPROCESS_TF, GammaTransform(0.14)]),
    "Gamma Dec AUC 0.62": transforms.Compose([PREPROCESS_TF, GammaTransform(0.12)]),
    "Gamma Dec AUC 0.58": transforms.Compose([PREPROCESS_TF, GammaTransform(0.10)]),
}

GAMMA_DEC_TF_EDGE = {
    "Gamma Dec Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Dec AUC 0.69": transforms.Compose([PREPROCESS_TF, GammaTransform(0.06)]),
    "Gamma Dec AUC 0.53": transforms.Compose([PREPROCESS_TF, GammaTransform(0.01)]),
}


MAGNIFY_TF = {
    "Magnify AUC 0.86": transforms.Compose(
        [
            PREPROCESS_TF,
        ]
    ),
    "Magnify AUC 0.82": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.6), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.78": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.7), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.74": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.81), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.70": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.92), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.66": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.05), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.62": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.25), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify AUC 0.58": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 2.5), antialias=True),
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
