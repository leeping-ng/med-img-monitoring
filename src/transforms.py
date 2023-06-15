from torchvision import transforms


preprocess_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
    ]
)
train_transforms = transforms.Compose(
    [
        preprocess_transforms,
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, (0.9, 1), antialias=True),
    ]
)


class ContrastTransform:
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, img):
        # output = gain*input + bias
        return (img * self.gain).clip(min=0.0, max=255.0)
