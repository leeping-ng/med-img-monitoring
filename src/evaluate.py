import pytorch_lightning as L
from rsna_dataloader import RSNAPneumoniaDataModule
# from model import LitModel
from config import load_config
from model import ResNetClassifier
from torchsummary import summary


CONFIG_PATH = "config.yml"

if __name__ == "__main__":
    configs = load_config(CONFIG_PATH)
    #model = LitModel.load_from_checkpoint("lightning_logs/version_33/checkpoints/epoch=19-step=12760.ckpt")
    model = ResNetClassifier.load_from_checkpoint("lightning_logs/version_88/checkpoints/epoch=18-step=12122.ckpt")
    summary(model, input_size=(3, 224, 224), batch_size=32)
    model.eval()
    rsna = RSNAPneumoniaDataModule()
    trainer = L.Trainer()
    trainer.test(model=model, dataloaders=rsna.test_dataloader())
    #trainer.predict(model=model, dataloaders=rsna.predict_dataloader())
    # print(model.matrix)