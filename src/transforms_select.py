from torchvision import transforms

from transforms_custom import ContrastTransform, SaltPepperNoiseTransform


PREPROCESS_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
    ]
)

TRAIN_TRANSFORMS = transforms.Compose(
    [
        PREPROCESS_TRANSFORMS,
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, (0.9, 1), antialias=True),
    ]
)


CONTRAST_TRANSFORMS = {
    "Contrast 180%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, ContrastTransform(1.8)]
    ),
    "Contrast 160%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, ContrastTransform(1.6)]
    ),
    "Contrast 140%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, ContrastTransform(1.4)]
    ),
    "Contrast 120%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, ContrastTransform(1.2)]
    ),
    "Contrast Unchanged": transforms.Compose(
        [PREPROCESS_TRANSFORMS, ContrastTransform(1.0)]
    ),
    "Contrast 80%": transforms.Compose([PREPROCESS_TRANSFORMS, ContrastTransform(0.8)]),
    "Contrast 60%": transforms.Compose([PREPROCESS_TRANSFORMS, ContrastTransform(0.6)]),
    "Contrast 40%": transforms.Compose([PREPROCESS_TRANSFORMS, ContrastTransform(0.4)]),
    "Contrast 20%": transforms.Compose([PREPROCESS_TRANSFORMS, ContrastTransform(0.2)]),
}

SALT_PEPPER_NOISE_TRANSFORMS = {
    "Salt Pepper Noise 0%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.0)]
    ),
    "Salt Pepper Noise 5%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.05)]
    ),
    "Salt Pepper Noise 10%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.1)]
    ),
    "Salt Pepper Noise 20%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.2)]
    ),
    "Salt Pepper Noise 30%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.3)]
    ),
    "Salt Pepper Noise 40%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.4)]
    ),
    "Salt Pepper Noise 50%": transforms.Compose(
        [PREPROCESS_TRANSFORMS, SaltPepperNoiseTransform(0.5)]
    ),
}
