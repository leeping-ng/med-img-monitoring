import pytorch_lightning as L
from rsna_dataloader import RSNAPneumoniaDataModule
# from model import LitModel
from resnet_model import ResNetClassifier
from torchsummary import summary

#model = LitModel.load_from_checkpoint("lightning_logs/version_33/checkpoints/epoch=19-step=12760.ckpt")
model = ResNetClassifier.load_from_checkpoint("lightning_logs/version_84/checkpoints/epoch=19-step=12760.ckpt")
summary(model, input_size=(3, 224, 224), batch_size=32)
model.eval()
rsna = RSNAPneumoniaDataModule()
trainer = L.Trainer()
trainer.test(model=model, dataloaders=rsna.test_dataloader())
#trainer.predict(model=model, dataloaders=rsna.predict_dataloader())
# print(model.matrix)