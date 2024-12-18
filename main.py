import argparse
import warnings
import numpy as np
import yaml
import numpy as np
from utils.data import get_data
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from torchmetrics import F1Score
import pdb
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=DeprecationWarning)

from model.sensor_attention import SensorAttention, PositionalEncoding , AttentionWithContext
from model.self_attention.encoder import EncoderLayer

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        # X: NumPy array of shape (num_samples, window_size, num_features)
        # y: NumPy array of shape (num_samples,)
        self.X = torch.tensor(X, dtype=torch.float32)  # convert to float tensor
        self.y = torch.tensor(y, dtype=torch.long)     # convert to long tensor (for class labels)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]   # shape: (window_size, num_features)
        label = self.y[idx]    # scalar indicating the class

        # Apply any optional transforms (if needed)
        if self.transform:
            sample = self.transform(sample)

        return sample, label


class TimeSeriesModule(pl.LightningModule):
    def __init__(self, input_size=18, num_classes=12, lr=1e-3,seq_len=33, d_model=128,rate=0.2):
        super(TimeSeriesModule, self).__init__()
        self.save_hyperparameters()
        self.sensor_attention = SensorAttention(n_filters=128, kernel_size=3, dilation_rate=2,embd=input_size)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=1)
        self.positional_encoding = PositionalEncoding(d_model=d_model,  max_len=seq_len)
        self.dropout = nn.Dropout(rate)
        self.AttentionWithContext = AttentionWithContext(feature_dim=d_model, bias=True, return_attention=False)
        self.encoder1 = EncoderLayer(d_model=d_model, num_heads=4, dff=512, rate=0.2)
        self.encoder2 = EncoderLayer(d_model=d_model, num_heads=4, dff=512, rate=0.2)
        self.fc1 = nn.Linear(d_model, num_classes*4)
        self.fc2 = nn.Linear(num_classes*4, num_classes)
        self.fc1_relu = nn.ReLU()
        self.d_model = d_model
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=num_classes, average='macro', task = 'multiclass')
        
    def forward(self, x):
        si, _ = self.sensor_attention(x)
        x = F.relu(self.conv1d(si.permute(0, 2, 1))).permute(0,2,1)  # Permute to match PyTorch's (batch_size, channels, sequence_length) format
        
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x= self.positional_encoding(x)
        x = self.dropout(x)
        
        x= self.encoder1(x)
        x= self.encoder2(x)
        x= self.AttentionWithContext(x)
        x= self.fc1_relu(self.fc1(x)) 
        x= self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        f1_train = self.f1_macro(preds, y)
        
        # Log F1 score
        self.log("train_f1_macro", f1_train, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        #self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        f1_val = self.f1_macro(preds, y)
        # Log F1 score
        self.log("val_f1_macro", f1_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=False)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        f1_test = self.f1_macro(preds, y)
        self.log("test_f1_macro", f1_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        #self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, factor=0.1, patience=4, min_lr=1e-4, verbose=True),
            'monitor': 'val_loss'  # Metric to monitor for ReduceLROnPlateau
        }
        return [optimizer], [scheduler]
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Attention Based HAR Model Training')

    parser.add_argument('--train', action='store_true', default=False, help='Training Mode')
    parser.add_argument('--test', action='store_true', default=False, help='Testing Mode')
    parser.add_argument('--epochs', default=100, type=int, help='Number of Epochs for Training')
    parser.add_argument('--dataset', default='pamap2', type=str, help='Name of Dataset for Model Training')
    parser.add_argument('--load', action='store_true', default=False, help='Load presaved data')
    parser.add_argument('--logger', action='store_true', default=False, help='Use Wandb Logger')


    args = parser.parse_args()

    model_config_file = open('configs/model.yaml', mode='r')
    model_cfg = yaml.load(model_config_file, Loader=yaml.FullLoader)

    if not args.load:
        train_x, train_y, val_x, val_y, test_x, test_y = get_data(dataset=args.dataset)

        np.savez_compressed(args.dataset+'_data.npz',
                    train_x=train_x, train_y=train_y,
                    val_x=val_x, val_y=val_y,
                    test_x=test_x, test_y=test_y)
        print("train_data_saved~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    else :
        print("loading saved data~~~~~~~~~~~~~~~")
        data = np.load(args.dataset+'_data.npz')
        train_x = data['train_x']
        train_y = data['train_y']
        val_x = data['val_x']
        val_y = data['val_y']
        test_x = data['test_x']
        test_y = data['test_y']

    
    bs= 256 #batch size

    train_dataset = TimeSeriesDataset(train_x,  train_y)
    val_dataset = TimeSeriesDataset(val_x, val_y)
    test_dataset = TimeSeriesDataset (test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)
    
    
    wandb_logger = WandbLogger(project="HAR" , name='Self-attention-ECAI')
    if args.dataset == 'opp':
        num_classes = 18
    elif args.dataset == 'pamap2':
        num_classes = 12    
    elif args.dataset == 'skoda':
        num_classes = 11
    else:
        num_classes = 12
    # 12 for pamap2 and uschad, 18 for opp. 11 for skoda

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint"
    )
    early_stopping = EarlyStopping(monitor="val_acc", patience=20, mode="max", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    num_sensors = next(iter(val_loader))[0].size(2)
    seq_len = next(iter(val_loader))[0].size(1)
    

    model = TimeSeriesModule(input_size=num_sensors, num_classes=num_classes, lr=0.001, seq_len=seq_len)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger if args.logger else None,
        callbacks=[checkpoint_callback,early_stopping,lr_monitor],
        accelerator='gpu',
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test using the best checkpoint
    trainer.test(model, test_loader)
    
    
    
    