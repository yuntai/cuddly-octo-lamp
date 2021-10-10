import os

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from tqdm import tqdm
from dataset import HaiDataset

from dataset import get_dataset
import score

# x -> z
class Encoder(nn.Module):
    def __init__(self, seq_len=100, input_features=33, hidden_size=100, latent_size=20, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lin = nn.Linear(in_features=seq_len*hidden_size*2, out_features=latent_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_size=20, seq_len=100, hidden_size=128, output_features=33, num_layers=1):
        super().__init__()
        self.lin = nn.Linear(in_features=latent_size, out_features=seq_len//2)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.upsample = nn.Upsample(scale_factor=2)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size*2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.output = nn.Linear(hidden_size*2, output_features)

    def forward(self, x):
        x = self.lin(x)
        x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)
        x = self.output(x)
        x = torch.tanh(x)
        return x


class CriticX(nn.Module):
    def __init__(self, dropout=0.25, seq_len=100, input_features=33, block_size=4, channel_size=64, alpha=0.2):
        super().__init__()

        kernel_size = 5
        x = kernel_size // 2 * 2

        layers = []
        in_channels = input_features
        for i in range(block_size):
            layers += [
                nn.Conv1d(in_channels, channel_size, kernel_size=5),
                nn.LeakyReLU(alpha),
                nn.Dropout(dropout),
            ]
            in_channels = channel_size

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        _input_dim = (seq_len - (kernel_size // 2) * 2 * block_size) * channel_size
        self.lin = nn.Linear(_input_dim, 1)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.transpose(1,2)
        x = self.layers(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x


class CriticZ(nn.Module):
    def __init__(self, hidden_size=100, alpha=0.2, dropout=0.2, latent_size=20):
        super(CriticZ, self).__init__()
        layers = [
            nn.Linear(in_features=latent_size, out_features=hidden_size),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TADGan(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.encoder = Encoder(
            seq_len=self.hparams.window_size,
            input_features=self.hparams.input_features,
            latent_size=self.hparams.latent_size
        )
        self.generator = Generator(
            output_features=self.hparams.input_features,
            latent_size=self.hparams.latent_size,
            seq_len=self.hparams.window_size
        )
        self.critic_x = CriticX(
            input_features=self.hparams.input_features,
            seq_len=self.hparams.window_size
        )
        self.critic_z = CriticZ(latent_size=self.hparams.latent_size)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        #x = self.proj(x)
        #x = torch.tanh(x)

        z = self.encoder(x)

        y_hat = self.generator(z)

        critic_score = self.critic_x(x)
        return x, y_hat, critic_score

    def compute_gradient_penalty(self, critic, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        __slice = (...,) + (None,) * (real_samples.dim() - 1)
        alpha = torch.rand(real_samples.size(0))[__slice].to(self.device)

        # get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = critic(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)

        # get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.contiguous()

        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_dataloader_nyc(self):
        from dataset import nyc_dataset
        ds = nyc_dataset(window_size=self.hparams.window_size)
        #ds = HaiDataset(vals, self.hparams.window_size)
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=12, shuffle=True)

    def train_dataloader(self):
        return self.train_dataloader_haicon()

    def val_dataloader(self):
        return self.train_dataloader_haicon()

    def train_dataloader_haicon(self):
        from dataset import get_dataset
        ds = get_dataset(self.hparams.window_size, ['C01','C03'])
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=12, shuffle=True)

    def validation_step(self, batch, batch_idx):
        x, x_hat, cirtic_score = self(batch)
        l2_loss = self.loss_fn(x, x_hat)
        self.log('val_l2', l2_loss, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch

        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_size)
        z = z.type_as(x)

        if optimizer_idx == 0:
            x_= self.generator(z)
            cx = self.critic_x(x)
            cx_ = self.critic_x(x_)
            gp_x = self.compute_gradient_penalty(self.critic_x, x, x_)

            cx_loss = cx_.mean() - cx.mean() + self.hparams.lambda_gp * gp_x
            self.log('cx', cx_loss, on_step=False, on_epoch=True, prog_bar=True)
            return cx_loss

        elif optimizer_idx == 1:
            z_ = self.encoder(x)
            cz = self.critic_z(z)
            cz_ = self.critic_z(z_)
            gp_z = self.compute_gradient_penalty(self.critic_z, z, z_)
            cz_loss = cz_.mean() - cz.mean() + self.hparams.lambda_gp * gp_z
            self.log('cz', cz_loss, on_step=False, on_epoch=True, prog_bar=True)
            return cz_loss

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        else:
            cx = self.critic_x(x)
            cx_ = self.critic_x(self.generator(z))

            cz = self.critic_z(z)
            x_ = self.encoder(x)
            cz_ = self.critic_z(x_)

            x_re = self.generator(x_)

            l2_loss = self.loss_fn(x, x_re)

            ge_loss = -cx_.mean()-cz_.mean() + self.hparams.lambda_recon * l2_loss

            self.log('ge', ge_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('l2', l2_loss, on_step=False, on_epoch=True, prog_bar=True)

            return ge_loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the loss
        items.pop("loss", None)
        return items

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        n_critic = self.hparams.n_critic

        params = list(self.critic_x.parameters())# + list(self.proj.parameters())
        opt_cx = torch.optim.Adam(params, lr=lr, betas=(b1, b2))
        params = list(self.critic_z.parameters())# + list(self.proj.parameters())
        opt_cz = torch.optim.Adam(params, lr=lr, betas=(b1, b2))

        ge_params = list(self.encoder.parameters()) + list(self.generator.parameters())
        opt_ge = torch.optim.Adam(ge_params, lr=lr, betas=(b1, b2))

        return (
            {'optimizer': opt_cx, 'frequency': n_critic},
            {'optimizer': opt_cz, 'frequency': n_critic},
            {'optimizer': opt_ge, 'frequency': 1}
        )

    def configure_callbacks(self):
        exp_name = self.trainer.logger.experiment.name
        #logger.info(f"{exp_name=}")
        checkpoint_cb = ModelCheckpoint(
            #dirpath=f"./res/mlm_base={self.hparams.base_model}&max_seq_len={self.hparams.max_seq_len}/{exp_name}",
            monitor="l2",
            save_top_k=10,
            filename='{epoch}-{step}-{l2:.3f}',
            mode='max'
        )
        #lr_monitor_cb = LearningRateMonitor(logging_interval='step')

        return [checkpoint_cb]

    def on_epoch_end(self):
        pass

def predict(ckpt, _type='val'):

    model = TADGan.load_from_checkpoint(ckpt).cuda()

    #from dataset import nyc_dataset
    #ds = nyc_dataset()

    ds = get_dataset(model.hparams.window_size, ['C01'], _type=_type)
    dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=256, drop_last=False)

    x, x_hat, critic_score = [], [], []
    with torch.no_grad():
        for b in tqdm(dl):
            x_, x_hat_, critic_score_ = model(b.cuda())
            x.append(x_.cpu())
            x_hat.append(x_hat_.cpu())
            critic_score.append(critic_score_.cpu())

    x = torch.cat(x, dim=0)
    x_hat = torch.cat(x_hat, dim=0)
    critic_score = torch.cat(critic_score, dim=0)

    anomaly_score, gt, pred = score.score_anomalies(x, x_hat, critic_score, rec_error_type="dtw", comb="mult")
    pred = pred.mean(dim=-1)

    return x, x_hat, critic_score, anomaly_score, gt, pred

def fit(args: Namespace) -> None:
    print(args)
    pl.seed_everything(args.seed)
    model = TADGan(**vars(args))
    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        logger=WandbLogger(project=args.project),
        amp_level='O2',
        precision=16,
        accelerator='ddp'
    )
    trainer.fit(model)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    p.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    p.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
    p.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    p.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    p.add_argument("--latent_size", type=int, default=20, help="dimensionality of the latent space")
    p.add_argument("--window_size", type=int, default=100, help="window size")
    p.add_argument("--input_features", type=int, default=2, help="number of input features")
    p.add_argument("--n_critic", type=int, default=5, help="n_critic")
    p.add_argument("--max_epochs", type=int, default=100, help="max_epochs")
    p.add_argument("--lambda_gp", type=float, default=10., help="gradient penalty weight")
    p.add_argument("--lambda_recon", type=float, default=10., help="reconstruction loss weight")
    p.add_argument("--seed", type=int, default=42, help="seed")
    p.add_argument('--project', type=str, default='dacon-haicon')

    hparams = p.parse_args()

    fit(hparams)
