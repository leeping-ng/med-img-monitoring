
import pytorch_lightning as L
from rsna_dataloader import RSNAPneumoniaDataModule
from model import LitModel

    
model = LitModel((1, 224, 224), 2, 32)
rsna = RSNAPneumoniaDataModule()
trainer = L.Trainer(max_epochs=1, log_every_n_steps=10)
trainer.fit(model, rsna)