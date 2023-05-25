import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, BinaryAUROC

# adapted from https://www.scaler.com/topics/pytorch/build-and-train-an-image-classification-model-with-pytorch-lightning/

class LitModel(pl.LightningModule):
    '''model architecture, training, testing and validation loops'''
    def __init__(self, input_shape, num_classes, batch_size, learning_rate=3e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # model architecture
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        
        n_sizes = self._get_output_shape(input_shape)

        # linear layers for classifier head
        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="binary", num_classes=2)
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.bcm = BinaryConfusionMatrix()
        self.b_auroc = BinaryAUROC()
        # confusion matrix
        self.matrix = torch.tensor([[0,0], [0,0]]).to("cuda:0")

    def _get_output_shape(self, shape):
        '''returns the size of the output tensor from the conv layers'''

        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._feature_extractor(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    
    # computations
    def _feature_extractor(self, x):
        '''extract features from the conv blocks'''
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    
    def forward(self, x):
        '''produce final model output'''
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
    # train loop
    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]

        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # metric
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_f1', f1, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        return loss
    
    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        self.log('val_f1', f1, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        b_auroc = self.b_auroc(preds, y)
        # accumulate to confusion matrix
        self.matrix += self.bcm(preds, y)
        self.log('test_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', acc, prog_bar=True, batch_size=self.batch_size)
        self.log('test_precision', precision, prog_bar=True, batch_size=self.batch_size)
        self.log('test_recall', recall, prog_bar=True, batch_size=self.batch_size)
        self.log('test_f1', f1, prog_bar=True, batch_size=self.batch_size)
        self.log('test_auc_roc', b_auroc, prog_bar=True, batch_size=self.batch_size)

        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch["image"], batch["target"]
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # use softmax for K-S test later
        print(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
    
    # optimizers 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



