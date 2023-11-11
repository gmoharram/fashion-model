import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    accelerator = "gpu"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    accelerator = "mps"
else:
    device = torch.device("cpu")

from main_code.constants import TARGET_CHANNELS

from IPython.core.debugger import set_trace


class FashionAutoEncoder(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        self.save_hyperparameters(hparams)  # save hyperparameters to tensorboard

        self.loss = nn.MSELoss(reduction="mean")

        resnet = torchvision.models.resnet18(
            pretrained=True
        )  # download pretrained resnet18 model
        resnet_used_layers = nn.Sequential(
            *list(resnet.children())[:-3]
        )  # remove last 3 layers from pretrained resnet18 model
        if hparams["num_resnet_trainable"] > 0:
            self.resnet_trainable = resnet_used_layers[
                -hparams["num_resnet_trainable"] :
            ]  # make last n layers of remaining resnet network trainable
            self.resnet_frozen = resnet_used_layers[: -hparams["num_resnet_trainable"]]
        else:
            self.resnet_trainable = nn.Identity()
            self.resnet_frozen = resnet_used_layers
        for param in self.resnet_frozen.parameters():  # freeze pretrained parameters
            param.requires_grad = False

        if hparams["num_batches"] == 1:
            norm_layer = nn.InstanceNorm1d
        else:
            norm_layer = nn.BatchNorm1d

        self.encoder_compress = nn.Sequential(
            nn.MaxPool2d(
                5,
                stride=3,
            ),
            nn.Conv2d(256, 16, kernel_size=1),
            nn.BatchNorm2d(
                16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(in_features=13 * 10 * 16, out_features=128, bias=False),
            norm_layer(num_features=128, momentum=0.1),
            nn.ReLU(True),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=13 * 10 * 16, bias=False),
            norm_layer(num_features=13 * 10 * 16, momentum=0.1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(  # define trainable "decoder" architecture
            nn.Conv2d(16, 8, kernel_size=1),
            nn.BatchNorm2d(
                8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, TARGET_CHANNELS, kernel_size=3, stride=1),
            nn.BatchNorm2d(
                TARGET_CHANNELS,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                TARGET_CHANNELS, TARGET_CHANNELS, kernel_size=(30, 15), stride=14
            ),
            nn.BatchNorm2d(
                TARGET_CHANNELS,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                TARGET_CHANNELS, TARGET_CHANNELS, kernel_size=(5, 6), stride=3
            ),
            nn.BatchNorm2d(
                TARGET_CHANNELS,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.resnet_trainable(self.resnet_frozen(x))
        x = self.encoder_compress(x)
        x = self.encoder_fc(x.view(x.size(0), -1))
        x = self.decoder_fc(x)
        x = self.decoder(x.view(x.size(0), 16, 13, 10))

        return x

    def general_step(self, batch, batch_idx, mode):
        inputs, targets = batch

        # forward pass
        outputs = self.forward(inputs)

        # loss
        loss = self.loss(outputs, targets)
        return loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        self.log("val_loss", avg_loss)

    def configure_optimizers(self):
        optim = None

        if self.hparams["optimizer"] == "Adam":
            optim = torch.optim.Adam(
                self.parameters(), lr=self.hparams["learning_rate"]
            )
        elif self.hparams["optimizer"] == "SGD":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["learning_rate"],
                momentum=self.hparams["momentum"],
            )

        return optim

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    @property
    def is_mps(self):
        """
        Check if model parameters are allocated on the MPS.
        """
        return next(self.parameters()).is_mps

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self, path)
