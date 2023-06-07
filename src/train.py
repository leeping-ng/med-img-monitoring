
import pytorch_lightning as pl
from torchsummary import summary
from torchvision import transforms

from config import load_config
from model import ResNetClassifier
from rsna_dataloader import RSNAPneumoniaDataModule


CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    model = ResNetClassifier(2, 18, tune_fc_only=False)
    model.cuda()
    summary(model, input_size=(3, 224, 224), batch_size=32)
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.Resize(256, antialias=True),
                                        transforms.RandomResizedCrop(224, (0.8,1), antialias=True)])
                                        # transforms.CenterCrop(224)])
    # Need transforms for validation?
    rsna = RSNAPneumoniaDataModule(configs, train_transforms=train_transforms, val_transforms=train_transforms)
    pl.seed_everything(33, workers=True)
    trainer = pl.Trainer(max_epochs=100, callbacks=[pl.callbacks.EarlyStopping(monitor="val_roc-auc", mode="max", patience=10)])
    trainer.fit(model, rsna)

    trainer.test(dataloaders=rsna.test_dataloader(), ckpt_path="best")