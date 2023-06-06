
import pytorch_lightning as L
from rsna_dataloader import RSNAPneumoniaDataModule
#from model import LitModel
from resnet_model import ResNetClassifier

    
#model = LitModel((1, 224, 224), 2, 32)
model = ResNetClassifier(2, 18)
rsna = RSNAPneumoniaDataModule()
L.seed_everything(33, workers=True)
trainer = L.Trainer(max_epochs=100, callbacks=[L.callbacks.EarlyStopping(monitor="val_roc-auc", mode="max", patience=10)])
trainer.fit(model, rsna)

trainer.test(dataloaders=rsna.test_dataloader(), ckpt_path="best")