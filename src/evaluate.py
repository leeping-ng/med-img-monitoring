import pytorch_lightning as L
from rsna_dataloader import RSNAPneumoniaDataModule
from model import LitModel
from torchsummary import summary

model = LitModel.load_from_checkpoint("lightning_logs/version_33/checkpoints/epoch=19-step=12760.ckpt")
summary(model, input_size=(1, 224, 224), batch_size=32)
model.eval()
rsna = RSNAPneumoniaDataModule()
trainer = L.Trainer()
trainer.test(model=model, dataloaders=rsna.test_dataloader())
#trainer.predict(model=model, dataloaders=rsna.predict_dataloader())
print(model.matrix)