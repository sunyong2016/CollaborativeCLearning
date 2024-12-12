from typing import Dict, Tuple

import pytorch_lightning as pl
from lightly.models.utils import (activate_requires_grad,
                                  deactivate_requires_grad)
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.dist import print_rank_zero
from lightly.utils.scheduler import CosineWarmupScheduler
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import SGD


class MLPClassifier(LightningModule):

    def __init__(
            self,
            model: Module,
            batch_size: int,
            feature_dim: int = 2048,
            num_classes: int = 1000,
            topk: Tuple[int, ...] = (1, 5),
            freeze_model: bool = False,
    ) -> None:
        super().__init__()
        # self.model = model
        self.backbone = model.backbone
        self.feature_dim = feature_dim
        self.batch_size = batch_size  # batch_size
        self.num_classes = num_classes
        self.topk = topk
        self.freeze_model = freeze_model

        # MLP Classification
        # self.classification_head = torch.nn.Sequential(
        #     torch.nn.Linear(feature_dim, 256),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256, num_classes),
        # )

        # LR Classification
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

    def forward(self, images: Tensor) -> Tensor:
        # features = self.model.forward(images)[1].flatten(start_dim=1)
        features = self.backbone(images).flatten(start_dim=1)
        return self.classification_head(features)

    def shared_step(self, batch,
                    batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log("train_loss", loss, batch_size=batch_size)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 sync_dist=True,
                 batch_size=batch_size)
        self.log_dict(log_dict,
                      prog_bar=True,
                      sync_dist=True,
                      batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            # parameters += self.model.parameters()
            parameters += self.backbone.parameters()
        optimizer = SGD(
            parameters,
            lr=0.1 * self.batch_size * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler":
            CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]


def MLP_eval(model, batch_size, num_workers, accelerator, devices, num_classes,
             linear_tr_loader, linear_te_loader, ft_max_epochs):
    # Setup training data.

    # Train linear classifier.
    metric_callback = MetricCallback()
    trainer = pl.Trainer(
        max_epochs=ft_max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
    )
    classifier = MLPClassifier(
        model=model,
        batch_size=batch_size,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        freeze_model=False,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=linear_tr_loader,
        val_dataloaders=linear_te_loader,
    )

    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(
            f"max classification {metric}: {max(metric_callback.val_metrics[metric])}"
        )

    return trainer.logged_metrics


def create_data_loader_eval(path_to_data, input_size, batch_size, num_workers, test_rate = 0.4):
    _transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(image_mean, image_std)
    ])

    rs_data_set = torchvision.datasets.ImageFolder(path_to_data, transform=_transform)
    len_rs_data = len(rs_data_set)            # total number of examples
    num_test    = int(test_rate * len_rs_data)  # take  10% for test
    num_train   = len_rs_data - num_test  # take  80% for train

    train_rs_data, test_rs_data = torch.utils.data.random_split(rs_data_set, lengths = [num_train, num_test])
    print(len_rs_data, num_train, len(train_rs_data))

    # create a dataloader for embedding
    rs_train_loader = torch.utils.data.DataLoader(
        train_rs_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    rs_test_loader = torch.utils.data.DataLoader(
        test_rs_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return rs_train_loader, rs_test_loader




def training_MLP(smog_model, args):
    rs_train_loader, rs_test_loader = create_data_loader_eval(
        args.input_size, args.batch_size, args.nThreads)

    smog_model.eval()
    logged_metrics = MLP_eval(smog_model, args.batch_size, args.nThreads,
                              accelerator, devices_num, num_classes,
                              rs_train_loader, rs_test_loader,
                              args.eval_max_epochs)
    return logged_metrics
