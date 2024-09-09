# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
)

import torch
import numpy as np
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)


class PropertyLoss(TaskLoss):
    def __init__(
        self,
        task_dim,
        type_map,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        beta: float = 1.00,
        coord_noise: float = 0.0,
        box_noise: float = 0.0,
        pert_prop: float = 1.0,
        **kwargs,
    ):
        r"""Construct a layer to compute loss on property.

        Parameters
        ----------
        task_dim : float
            The output dimension of property fitting net.
        loss_func : str
            The loss function, such as "smooth_mae", "mae", "rmse".
        metric : list
            The metric such as mae, rmse which will be printed.
        beta:
            The 'beta' parameter in 'smooth_mae' loss.
        """
        super().__init__()
        self.task_dim = task_dim
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.coord_noise = coord_noise
        self.box_noise = box_noise
        self.type_map = type_map
        self.pert_prop = pert_prop

    def forward(self, input_dict, model, label, natoms, learning_rate=0.0, mae=False):
        """Return loss on properties .

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        if (self.coord_noise > 0) or (self.box_noise > 0):
            import dpdata
            nloc = input_dict["atype"].shape[1]
            nbz = input_dict["atype"].shape[0]
            mask_num = nloc # need to discuss
            new_coord = []
            new_box = []
            for ii in range(nbz):
                # 提取每个system
                frame_data = {}
                frame_data["orig"] = np.array([0, 0, 0])
                frame_data['atom_names'] = self.type_map
                frame_data['atom_numbs'] = np.zeros(len(self.type_map),dtype=int).tolist()
                frame_data['atom_types'] = input_dict["atype"][ii].cpu().numpy()
                for jj in frame_data['atom_types']:
                    frame_data['atom_numbs'][jj] += 1
                frame_data['cells'] = np.array([input_dict["box"][ii].reshape([3,3]).tolist()])
                frame_data['coords'] = np.array([input_dict["coord"][ii].tolist()])
                frame = dpdata.System(data=frame_data, type_map=self.type_map)
                # 加扰动
                new_frame = frame.perturb(pert_num=1,atom_pert_distance=self.coord_noise,cell_pert_fraction=self.box_noise,atom_pert_prob=self.pert_prop)
                # 转化成input_dict
                # dict_keys(['coord', 'atype', 'box', 'do_atomic_virial', 'fparam', 'aparam'])
                # input_dict["coord"] : device='cuda:0' dtype=torch.float64 torch.Size([8, 4, 3])
                # input_dict["box"]:  dtype=torch.float64  torch.Size([8, 9])
                # new_frame["cells"]: numpy (1, 3, 3)
                # new_frame["coords"]: numpy (1, 4, 3)
                new_coord.append(new_frame["coords"].reshape(-1,3))
                new_box.append(new_frame["cells"].reshape(9))
            new_coord = np.array(new_coord)
            new_box = np.array(new_box)
            # new_coord: 8,4,3
            # new_box: 8,9
            input_dict["coord"] = torch.tensor(new_coord, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
            input_dict["box"] = torch.tensor(new_box, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        model_pred = model(**input_dict)
        assert label["property"].shape[-1] == self.task_dim
        assert model_pred["property"].shape[-1] == self.task_dim
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        # loss
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(
                label["property"],
                model_pred["property"],
                reduction="sum",
                beta=self.beta,
            )
        elif self.loss_func == "mae":
            loss += F.l1_loss(
                label["property"], model_pred["property"], reduction="sum"
            )
        elif self.loss_func == "mse":
            loss += F.mse_loss(
                label["property"],
                model_pred["property"],
                reduction="sum",
            )
        elif self.loss_func == "rmse":
            loss += torch.sqrt(
                F.mse_loss(
                    label["property"],
                    model_pred["property"],
                    reduction="mean",
                )
            )
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        # more loss
        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(
                label["property"],
                model_pred["property"],
                reduction="mean",
                beta=self.beta,
            ).detach()
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(
                label["property"],
                model_pred["property"],
                reduction="mean",
            ).detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(
                label["property"],
                model_pred["property"],
                reduction="mean",
            ).detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(
                    label["property"],
                    model_pred["property"],
                    reduction="mean",
                )
            ).detach()

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "property",
                ndof=self.task_dim,
                atomic=False,
                must=False,
                high_prec=True,
            )
        )
        return label_requirement
