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
    "Blur 3x3": transforms.Compose([PREPROCESS_TF, BlurTransform(3)]),
    "Blur 7x7": transforms.Compose([PREPROCESS_TF, BlurTransform(7)]),
    "Blur 15x15": transforms.Compose([PREPROCESS_TF, BlurTransform(15)]),
    "Blur 31x31": transforms.Compose([PREPROCESS_TF, BlurTransform(31)]),
}

BLUR_TF_EDGE = {
    "Blur Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Blur AUC 0.69": transforms.Compose([PREPROCESS_TF, BlurTransform(107)]),
    "Blur AUC 0.67": transforms.Compose([PREPROCESS_TF, BlurTransform(113)]),
}

SHARPEN_TF = {
    "Sharpen Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Sharpen 6.25": transforms.Compose([PREPROCESS_TF, SharpenTransform(6.25)]),
    "Sharpen 12.5": transforms.Compose([PREPROCESS_TF, SharpenTransform(12.5)]),
    "Sharpen 25.0": transforms.Compose([PREPROCESS_TF, SharpenTransform(25.0)]),
    "Sharpen 50.0": transforms.Compose([PREPROCESS_TF, SharpenTransform(50.0)]),
}

SHARPEN_TF_EDGE = {
    "Sharpen Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Sharpen AUC 0.69": transforms.Compose([PREPROCESS_TF, SharpenTransform(150)]),
    "Sharpen AUC 0.66": transforms.Compose([PREPROCESS_TF, SharpenTransform(255)]),
}

SALT_PEPPER_NOISE_TF = {
    "Salt Pepper Noise Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Salt Pepper Noise 0.0125": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.0125)]
    ),
    "Salt Pepper Noise 0.025": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.025)]
    ),
    "Salt Pepper Noise 0.05": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.05)]
    ),
    "Salt Pepper Noise 0.1": transforms.Compose(
        [PREPROCESS_TF, SaltPepperNoiseTransform(0.1)]
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
    "Speckle Noise 0.025": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.025)]
    ),
    "Speckle Noise 0.05": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.05)]
    ),
    "Speckle Noise 0.1": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.1)]
    ),
    "Speckle Noise 0.2": transforms.Compose(
        [PREPROCESS_TF, SpeckleNoiseTransform(0.2)]
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
    "Contrast Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Inc 1.375": transforms.Compose([PREPROCESS_TF, ContrastTransform(1.375)]),
    "Contrast Inc 1.75": transforms.Compose([PREPROCESS_TF, ContrastTransform(1.75)]),
    "Contrast Inc 2.5": transforms.Compose([PREPROCESS_TF, ContrastTransform(2.5)]),
    "Contrast Inc 4.0": transforms.Compose([PREPROCESS_TF, ContrastTransform(4.0)]),
}

CONTRAST_INC_TF_EDGE = {
    "Contrast Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Inc AUC 0.69": transforms.Compose([PREPROCESS_TF, ContrastTransform(7)]),
    "Contrast Inc AUC 0.63": transforms.Compose([PREPROCESS_TF, ContrastTransform(80)]),
}

CONTRAST_DEC_TF = {
    "Contrast Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Dec 0.73": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.73)]),
    "Contrast Dec 0.57": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.57)]),
    "Contrast Dec 0.4": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.4)]),
    "Contrast Dec 0.25": transforms.Compose([PREPROCESS_TF, ContrastTransform(0.25)]),
}

CONTRAST_DEC_TF_EDGE = {
    "Contrast Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Contrast Dec AUC 0.69": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.085)]
    ),
    "Contrast Dec AUC 0.5": transforms.Compose(
        [PREPROCESS_TF, ContrastTransform(0.03)]
    ),
}

GAMMA_INC_TF = {
    "Gamma Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Inc 1.25": transforms.Compose([PREPROCESS_TF, GammaTransform(1.25)]),
    "Gamma Inc 2.0": transforms.Compose([PREPROCESS_TF, GammaTransform(2.0)]),
    "Gamma Inc 2.5": transforms.Compose([PREPROCESS_TF, GammaTransform(2.5)]),
    "Gamma Inc 3.0": transforms.Compose([PREPROCESS_TF, GammaTransform(3.0)]),
    "Gamma Inc 4.0": transforms.Compose([PREPROCESS_TF, GammaTransform(4.0)]),
}

GAMMA_INC_TF_EDGE = {
    "Gamma Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Inc AUC 0.69": transforms.Compose([PREPROCESS_TF, GammaTransform(6.9)]),
    "Gamma Inc AUC 0.5": transforms.Compose([PREPROCESS_TF, GammaTransform(12)]),
}


GAMMA_DEC_TF = {
    "Gamma Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Dec 0.66": transforms.Compose([PREPROCESS_TF, GammaTransform(0.66)]),
    "Gamma Dec 0.5": transforms.Compose([PREPROCESS_TF, GammaTransform(0.5)]),
    "Gamma Dec 0.25": transforms.Compose([PREPROCESS_TF, GammaTransform(0.25)]),
    "Gamma Dec 0.15": transforms.Compose([PREPROCESS_TF, GammaTransform(0.15)]),
}

GAMMA_DEC_TF_EDGE = {
    "Gamma Unchanged": transforms.Compose([PREPROCESS_TF]),
    "Gamma Dec AUC 0.69": transforms.Compose([PREPROCESS_TF, GammaTransform(0.06)]),
    "Gamma Dec AUC 0.53": transforms.Compose([PREPROCESS_TF, GammaTransform(0.01)]),
}


MAGNIFY_TF = {
    "Magnify Unchanged": transforms.Compose(
        [
            PREPROCESS_TF,
        ]
    ),
    "Magnify 6.25%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.0625), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 12.5%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.125), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 25%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.25), antialias=True),
            transforms.CenterCrop(224),
        ]
    ),
    "Magnify 50%": transforms.Compose(
        [
            PREPROCESS_TF,
            transforms.Resize(int(224 * 1.5), antialias=True),
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
