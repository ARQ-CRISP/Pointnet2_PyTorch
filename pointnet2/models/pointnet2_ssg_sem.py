import torch.nn as nn
import torch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
import pytorch_lightning as pl
from models.pointnet2_ssg_cls import BNMomentumScheduler
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_sched
import sys

sys.path.append('/home/shadowop/Documents/CRISP_repos/pointcloud_processing')

from torch_dataset import PointCloudGraspingDataset


class PointNet2SemSegSSG(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[0, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 0, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 9, kernel_size=1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        # print(f"Dimension of xyz: {xyz.shape}")
        # print(f"Dimension of features: {features}")
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            print(f"=========For {i}:")
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            print("It is OK====================")
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0]).transpose(1, 2)

    def training_step(self, batch, batch_idx):
        pc, labels = batch

        prediction = self.forward(pc).double()
        loss = nn.functional.mse_loss(prediction[..., 0], labels[...,  0])
        # with torch.no_grad():
        #     acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        log = dict(train_loss=loss)

        # return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        pc, labels = batch

        prediction = self.forward(pc).double()
        loss = nn.functional.mse_loss(prediction[..., 0], labels[...,  0])
        # acc = (torch.argmax(prediction, dim=1) == labels).float().mean()

        return dict(val_loss=loss)

    def configure_optimizers(self):
        def lr_lbmd(_): return max(
            0.5
            ** (
                int(
                    self.global_step
                    * self.batch_size
                    / 3e5
                )
            ),
            1e-5 / 1e-3,
        )

        def bn_lbmd(_): return max(
            0.5
            * 0.5
            ** (
                int(
                    self.global_step
                    * self.batch_size
                    / 3e5
                )
            ),
            1e-2,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=0.0,
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def prepare_data(self):
        dataset = PointCloudGraspingDataset("/home/shadowop/Documents/poc_grasping_from_pcd/test/test.h5")
        total = len(dataset)
        val_size = int(0.15*total)
        self.train_dset, self.val_dset = random_split(dataset, [total-val_size, val_size])

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.batch_size,
            shuffle=mode == "train",
            num_workers=8,
            pin_memory=True
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
