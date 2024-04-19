# SPDX-License-Identifier: LGPL-3.0-or-later
import torch
import numpy as np
from typing import (
    List,
)
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
from IPython import embed

class DenoiseLoss(TaskLoss):
    def __init__(
        self,
        ntypes,
        noise_type: str = "uniform",
        noise: float = 1.0,
        noise_mode: str = "fix_num",
        mask_num: int = 1,
        mask_prob: float = 0.15,
        mask_coord: bool = True,
        mask_type: bool = False,
        same_mask: bool = False,
        masked_token_loss=1.0,
        masked_coord_loss=1.0,
        beta=1.00,
        mask_loss_coord=True,
        mask_loss_token=True,
        max_fail_num=10,
        **kwargs,
    ):
        """Construct a layer to compute loss on coord, and type reconstruction."""
        super().__init__()
        self.ntypes = ntypes
        self.noise_type = noise_type
        self.noise = noise
        self.noise_mode = noise_mode
        self.mask_num = mask_num
        self.mask_prob = mask_prob
        self.same_mask = same_mask
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.has_coord = self.masked_coord_loss > 0.0
        self.has_token = self.masked_token_loss > 0.0
        self.beta = beta
        self.frac_beta = 1.00 / self.beta
        self.mask_loss_coord = mask_loss_coord
        self.mask_loss_token = mask_loss_token
        self.max_fail_num = max_fail_num
        self.mask_coord = mask_coord
        self.mask_type = mask_type


    def forward(self, input_dict, model, label, natoms, learning_rate=0.0, mae=False):
        """Return loss on coord and type denoise.

        Returns
        -------
        - loss: Loss to minimize.
        """
        label["clean_type"] = input_dict["atype"].clone().detach()
        label["clean_coord"] = input_dict["coord"].clone().detach()
        nloc = label["clean_type"].shape[1]
        assert nloc == input_dict["coord"].shape[1]
        #for i in range(self.max_fail_num):
        mask_num = 0
        if self.noise_mode == "fix_num":
            mask_num = self.mask_num
            if(nloc < mask_num):
                mask_num = nloc
        elif self.noise_mode == "prob":
            mask_num = int(self.mask_prob * nloc)
            if mask_num == 0:
                mask_num = 1
        else:
            NotImplementedError(f"Unknown noise mode {self.noise_mode}!")
        coord_mask_res = np.random.choice(range(nloc), mask_num, replace=False).tolist()
        coord_mask = np.isin(range(nloc), coord_mask_res) # nloc
        if self.same_mask:
            type_mask = coord_mask.copy()
        else:
            type_mask_res = np.random.choice(range(nloc), mask_num, replace=False).tolist()
            type_mask = np.isin(range(nloc), type_mask_res) #nloc

        # add noise for coord
        if self.mask_coord:
            noise_on_coord = 0.0
            if self.noise_type == "trunc_normal":
                noise_on_coord = np.clip(
                    np.random.randn(mask_num, 3) * self.noise,
                    a_min=-self.noise * 2.0,
                    a_max=self.noise * 2.0,
                )
            elif self.noise_type == "normal":
                noise_on_coord = np.random.randn(mask_num, 3) * self.noise
            elif self.noise_type == "uniform":
                noise_on_coord = np.random.uniform(
                    low=-self.noise, high=self.noise, size=(mask_num, 3)
                )
            else:
                NotImplementedError(f"Unknown noise type {self.noise_type}!")
            noise_on_coord = torch.tensor(noise_on_coord, device=env.DEVICE).detach() # mask_num 3
            input_dict["coord"][: ,coord_mask ,:] += noise_on_coord # nbz mask_num 3 // 
            label['coord_mask'] = torch.tensor(coord_mask, dtype=torch.bool, device=env.DEVICE) # 17
            label_updated_coord = (label["clean_coord"] - input_dict["coord"]).clone().detach()
            assert label_updated_coord[:,coord_mask].allclose(-1.00 * noise_on_coord)
        else:
            label['coord_mask'] = torch.tensor(np.zeros_like(coord_mask, dtype=bool), 
                                                dtype=torch.bool,
                                                device=env.DEVICE)

        # add mask for type
        if self.mask_type:
            input_dict["atype"][:,type_mask] = self.ntypes - 1
            label['type_mask'] = torch.tensor(type_mask, dtype=torch.bool, device=env.DEVICE)
        else:
            label['type_mask'] = torch.tensor(np.zeros_like(type_mask, dtype=bool),
                                                dtype=torch.bool,
                                                device=env.DEVICE)
        '''
        if self.pbc:
            _coord = normalize_coord(noised_coord, region, nloc)
        else:
            _coord = noised_coord.clone()
        try:
            nlist, nlist_loc, nlist_type, shift, mapping = make_env_mat(_coord, masked_type, region,
                                                                        rcut, sec,
                                                                        pbc=self.pbc,
                                                                        type_split=self.type_split,
                                                                        min_check=True)
        except RuntimeError as e:
            if i == self.max_fail_num - 1:
                RuntimeError(f"Add noise times beyond max tries {self.max_fail_num}!")
            continue
        '''
        model_pred = model(**input_dict)
        updated_coord = model_pred["updated_coord"]
        logits = model_pred["logits"]
        clean_type = label["clean_type"]
        coord_mask = label["coord_mask"]
        type_mask = label["type_mask"]

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        if self.has_coord:
            if self.mask_loss_coord:
                masked_updated_coord = updated_coord[:, coord_mask]
                masked_label_updated_coord = label_updated_coord[:, coord_mask]
                if masked_updated_coord.size(0) > 0:
                    coord_loss = F.smooth_l1_loss(
                        masked_updated_coord.view(-1, 3),
                        masked_label_updated_coord.view(-1, 3),
                        reduction="mean",
                        beta=self.beta,
                    )
                else:
                    coord_loss = torch.zeros(
                        1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
                    )[0]
            else:
                coord_loss = F.smooth_l1_loss(
                    updated_coord.view(-1, 3),
                    label_updated_coord.view(-1, 3),
                    reduction="mean",
                    beta=self.beta,
                )
            loss += self.masked_coord_loss * coord_loss
            more_loss["coord_l1_error"] = coord_loss.detach()
        if self.has_token:
            if self.mask_loss_token:
                masked_logits = logits[:, type_mask].view(-1, self.ntypes - 1)
                masked_target = clean_type[:, type_mask].view(-1)
                if masked_logits.size(0) > 0:
                    token_loss = F.nll_loss(
                        F.log_softmax(masked_logits, dim=-1),
                        masked_target.to(torch.int64),
                        reduction="mean",
                    )
                else:
                    token_loss = torch.zeros(
                        1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
                    )[0]
            else:
                token_loss = F.nll_loss(
                    F.log_softmax(logits.view(-1, self.ntypes - 1), dim=-1),
                    clean_type.view(-1).to(torch.int64),
                    reduction="mean",
                )
            loss += self.masked_token_loss * token_loss
            more_loss["token_error"] = token_loss.detach()

        return model_pred, loss, more_loss
    
    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "updated_coord",
                ndof=3,
                atomic=True,
                must=False,
                high_prec=True,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "logits",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=True,
            )
        )
        return label_requirement
