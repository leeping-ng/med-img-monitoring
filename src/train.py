
import pytorch_lightning as L
from rsna_dataloader import RSNAPneumoniaDataModule
from model import LitModel
from torchsummary import summary

    
model = LitModel((1, 224, 224), 2, 32)
summary(model, input_size=(1, 224, 224), batch_size=32)
rsna = RSNAPneumoniaDataModule()
L.seed_everything(33, workers=True)
trainer = L.Trainer(max_epochs=20, deterministic=True)
trainer.fit(model, rsna)

trainer.test(dataloaders=rsna.test_dataloader(), ckpt_path="best")