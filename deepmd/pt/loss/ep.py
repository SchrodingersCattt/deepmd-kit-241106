# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
    LOCAL_RANK,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
import numpy as np
from IPython import embed
import os
import logging
log = logging.getLogger(__name__)

def custom_huber_loss(predictions, targets, delta=1.0):
    error = targets - predictions
    abs_error = torch.abs(error)
    quadratic_loss = 0.5 * torch.pow(error, 2)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = torch.where(abs_error <= delta, quadratic_loss, linear_loss)
    return torch.mean(loss)


class EPLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate=1.0,
        start_pref_e=0.0,
        limit_pref_e=0.0,
        start_pref_f=0.0,
        limit_pref_f=0.0,
        start_pref_v=0.0,
        limit_pref_v=0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        use_l1_all: bool = False,
        inference=False,
        use_huber=False,
        huber_delta=1.0,
        noise_type: str = "uniform",
        noise: float = 0.2,
        noise_mode: str = "fix_num",
        mask_num: int = 1,
        mask_prob: float = 0.15,
        mask_coord: bool = True,
        mask_box: bool = False,
        has_fr_virial: bool = False,
        has_pert_virial: bool = False,
        has_box_ener: bool = False,
        sum_zero: bool = True,
        max_fail_num: int = 100000,
        k: float = 10.0,
        A: float = 1.0,
        **kwargs,
    ):
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        starter_learning_rate : float
            The learning rate at the start of the training.
        start_pref_e : float
            The prefactor of energsy loss at the start of the training.
        limit_pref_e : float
            The prefactor of energy loss at the end of the training.
        start_pref_f : float
            The prefactor of force loss at the start of the training.
        limit_pref_f : float
            The prefactor of force loss at the end of the training.
        start_pref_v : float
            The prefactor of virial loss at the start of the training.
        limit_pref_v : float
            The prefactor of virial loss at the end of the training.
        start_pref_ae : float
            The prefactor of atomic energy loss at the start of the training.
        limit_pref_ae : float
            The prefactor of atomic energy loss at the end of the training.
        start_pref_pf : float
            The prefactor of atomic prefactor force loss at the start of the training.
        limit_pref_pf : float
            The prefactor of atomic prefactor force loss at the end of the training.
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_f = (start_pref_f != 0.0 and limit_pref_f != 0.0) or inference
        self.has_v = (start_pref_v != 0.0 and limit_pref_v != 0.0) or inference

        # TODO EnergyStdLoss need support for atomic energy and atomic pref
        self.has_ae = (start_pref_ae != 0.0 and limit_pref_ae != 0.0) or inference
        self.has_pf = (start_pref_pf != 0.0 and limit_pref_pf != 0.0) or inference

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.huber = use_huber
        self.huber_delta = huber_delta
        self.use_l1_all = use_l1_all
        self.inference = inference

        self.noise_type = noise_type
        self.noise = noise
        self.noise_mode = noise_mode
        self.mask_num = mask_num
        self.mask_prob = mask_prob
        self.mask_coord = mask_coord
        self.mask_box = mask_box
        self.has_fr_virial = has_fr_virial
        self.has_pert_virial = has_pert_virial
        self.has_box_ener = has_box_ener
        self.sum_zero = sum_zero
        self.k = k
        self.A = A
        self.max_fail_num = max_fail_num

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        """Return loss on energy and force.

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

        # input_dict:
        # coord: nbz nloc 3
        # atype: nbz nloc 不动
        # box: nbz 9 不动
        
        nloc = input_dict["atype"].shape[1]
        nbz = input_dict["atype"].shape[0]
        label["clean_coord"] = input_dict["coord"].clone().detach()
        if self.mask_box:
            label["clean_box"] = input_dict["box"].clone().detach()
        
        # 将x加noise，并更新label['force']
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
        
        if self.mask_coord:
            noise_on_coord_all = torch.zeros(input_dict["coord"].shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) 
            coord_mask_all = torch.zeros(input_dict["atype"].shape, dtype=torch.bool, device=env.DEVICE) 
            for ii in range(nbz):
                # 对于每个batch单独处理
                noise_on_coord = 0.0
                coord_mask_res = np.random.choice(range(nloc), mask_num, replace=False).tolist()
                coord_mask = np.isin(range(nloc), coord_mask_res) # nloc
                if self.noise_type == "uniform":
                    if self.sum_zero:
                        for i in range(self.max_fail_num):
                            noise_on_coord = np.random.uniform(
                                low=-self.noise, high=self.noise, size=(3, mask_num - 1)
                            )
                            noise_of_last_atom = -1 * np.sum(noise_on_coord,axis=1).reshape(-1,1) #保证质心不变
                            if np.max(noise_of_last_atom)<self.noise and np.min(noise_of_last_atom)>-self.noise:
                                break
                            #if i == self.max_fail_num - 1:
                            #    raise RuntimeError(f"Add noise times beyond max tries {self.max_fail_num}!")

                        noise_on_coord = np.concatenate((noise_on_coord, noise_of_last_atom),axis=1).T
                        if i == self.max_fail_num - 1:
                            del noise_on_coord
                            noise_on_coord = np.random.uniform(
                                low=-self.noise, high=self.noise, size=(mask_num, 3)
                            )
                    else:
                        noise_on_coord = np.random.uniform(
                            low=-self.noise, high=self.noise, size=(mask_num, 3)
                        )
                else:
                    NotImplementedError(f"Unknown noise type {self.noise_type}!")
                
                noise_on_coord = torch.tensor(noise_on_coord, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) # mask_num 3
                input_dict["coord"][ii][coord_mask ,:] += noise_on_coord # nbz mask_num 3 //                 
                noise_on_coord = noise_on_coord.detach()

                noise_on_coord_all[ii] = noise_on_coord
                coord_mask_all[ii] = torch.tensor(coord_mask, dtype=torch.bool, device=env.DEVICE)
            label['coord_mask'] = coord_mask_all # 17
            f_ori = label.pop("force")
            del f_ori
            label["force"] = self.k * (label["clean_coord"] - input_dict["coord"]).clone().detach()
            corr_e_fpart = 0.5 * self.k * torch.sum(torch.sum(torch.square(noise_on_coord_all),axis=-1),axis=-1).view(nbz,1)
            assert label["force"][coord_mask_all].view(nbz,nloc,3).allclose(-1.00 * noise_on_coord_all.view(nbz,nloc,3) * self.k )
        
        else:           
            NotImplementedError(f"One must mask coord in easy potential mode!")
        
        corr_e_bpart = torch.zeros((nbz,1), dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) 
        if self.mask_box:
            noise_on_box_all = torch.zeros(input_dict["box"].shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) 
            for ii in range(nbz):
                # 对于每个batch单独处理
                noise_on_box = 0.0
                if self.noise_type == "uniform":
                    if self.sum_zero:
                        for i in range(self.max_fail_num):
                            noise_on_box = np.random.uniform(
                                low=-self.noise, high=self.noise, size=(3, 2)
                            )
                            noise_of_last_component = -1 * np.sum(noise_on_box, axis=1).reshape(-1, 1) #保证质心不变
                            if np.max(noise_of_last_component)<self.noise and np.min(noise_of_last_component)>-self.noise:
                                break

                        noise_on_box = np.concatenate((noise_on_box, noise_of_last_component),axis=1).T
                    else:
                        noise_on_box = np.random.uniform(
                            low=-self.noise, high=self.noise, size=(3, 3)
                        )
                else:
                    NotImplementedError(f"Unknown noise type {self.noise_type}!")
                
                noise_on_box = torch.tensor(noise_on_box, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) # 3 3
                # input_dict["box"] 现在是 [4, 9], input_dict["box"][ii]是[9], noise_on_box是[3,3]
                input_dict["box"][ii] += noise_on_box.reshape(-1).cpu()
                noise_on_box = noise_on_box.detach()
                noise_on_box_all[ii] = noise_on_box.reshape(9)
        
        
        
        np.savez(f"record_{LOCAL_RANK}.npz", coord = input_dict["coord"].cpu().detach().numpy(),
                               atype = input_dict["atype"].cpu().detach().numpy(),
                               box = input_dict["box"].cpu().detach().numpy(),
                               energy = label["energy"].cpu().detach().numpy(),
                               clean_coord = label["clean_coord"].cpu().detach().numpy(),
                               force = label["force"].cpu().detach().numpy())

        # 造virial
        fr_virial = torch.zeros((nbz, 9), dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE) 
        if self.has_fr_virial:
            atomic_virial = label["force"].unsqueeze(-1) @ input_dict["coord"].unsqueeze(-2)
            atomic_virial = atomic_virial.view(list(atomic_virial.shape[:-2]) + [9])
            fr_virial = torch.sum(atomic_virial,axis=1)
            assert fr_virial.shape[0]==nbz and fr_virial.shape[1]==9
        # TODO: 
        pert_virial = torch.zeros((nbz, 9), dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        if self.has_pert_virial:
            # pert_virial = A(h_star - h)@h.T
            delta_h = ((label["clean_box"] - input_dict["box"]).reshape(nbz,3,3)).transpose(-1,-2)
            assert delta_h.shape == (nbz, 3, 3)
            pert_virial = delta_h @ input_dict["box"].reshape(nbz, 3, 3)
            assert pert_virial.shape == (nbz, 3, 3)
            pert_virial = (pert_virial * self.A).reshape(-1,9)
        
        v_ori = label.pop("virial")
        del v_ori
        label["virial"] = fr_virial + pert_virial.to(device = env.DEVICE)
        label["find_virial"] = 1

        # 造能量
        if self.has_box_ener:
            delta_h = (input_dict["box"] - label["clean_box"]).reshape(nbz,3,3)
            embed()
            import sys
            sys.exit()
            #corr_e_bpart = 
        label["energy"] = label["energy"] + corr_e_fpart + corr_e_bpart

        model_pred = model(**input_dict)

        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            if not self.use_l1_all:
                l2_ener_loss = torch.mean(
                    torch.square(model_pred["energy"] - label["energy"])
                )
                if not self.inference:
                    more_loss["l2_ener_loss"] = self.display_if_exist(
                        l2_ener_loss.detach(), find_energy
                    )
                if not self.huber:
                    loss += atom_norm * (pref_e * l2_ener_loss)
                else:
                    l_huber_loss = custom_huber_loss(
                        model_pred["energy"], label["energy"], delta=self.huber_delta
                    )
                    loss += atom_norm * (pref_e * l_huber_loss)
                rmse_e = l2_ener_loss.sqrt() * atom_norm
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e.detach(), find_energy
                )
                # more_loss['log_keys'].append('rmse_e')
            else:  # use l1 and for all atoms
                l1_ener_loss = F.l1_loss(
                    model_pred["energy"].reshape(-1),
                    label["energy"].reshape(-1),
                    reduction="sum",
                )
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    F.l1_loss(
                        model_pred["energy"].reshape(-1),
                        label["energy"].reshape(-1),
                        reduction="mean",
                    ).detach(),
                    find_energy,
                )
                # more_loss['log_keys'].append('rmse_e')
            if mae:
                mae_e = (
                    torch.mean(torch.abs(model_pred["energy"] - label["energy"]))
                    * atom_norm
                )
                more_loss["mae_e"] = self.display_if_exist(mae_e.detach(), find_energy)
                mae_e_all = torch.mean(
                    torch.abs(model_pred["energy"] - label["energy"])
                )
                more_loss["mae_e_all"] = self.display_if_exist(
                    mae_e_all.detach(), find_energy
                )

        if self.has_f and "force" in model_pred and "force" in label:
            find_force = label.get("find_force", 0.0)
            pref_f = pref_f * find_force
            if "force_target_mask" in model_pred:
                force_target_mask = model_pred["force_target_mask"]
            else:
                force_target_mask = None
            if not self.use_l1_all:
                if force_target_mask is not None:
                    diff_f = (label["force"] - model_pred["force"]) * force_target_mask
                    force_cnt = force_target_mask.squeeze(-1).sum(-1)
                    l2_force_loss = torch.mean(
                        torch.square(diff_f).mean(-1).sum(-1) / force_cnt
                    )
                else:
                    diff_f = label["force"] - model_pred["force"]
                    l2_force_loss = torch.mean(torch.square(diff_f))
                if not self.inference:
                    more_loss["l2_force_loss"] = self.display_if_exist(
                        l2_force_loss.detach(), find_force
                    )
                if not self.huber:
                    loss += (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                else:
                    l_huber_loss = custom_huber_loss(
                        model_pred["force"], label["force"], delta=self.huber_delta
                    )
                    loss += (pref_f * l_huber_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_f = l2_force_loss.sqrt()
                more_loss["rmse_f"] = self.display_if_exist(rmse_f.detach(), find_force)
            else:
                l1_force_loss = F.l1_loss(
                    label["force"], model_pred["force"], reduction="none"
                )
                if force_target_mask is not None:
                    l1_force_loss *= force_target_mask
                    force_cnt = force_target_mask.squeeze(-1).sum(-1)
                    more_loss["mae_f"] = self.display_if_exist(
                        (l1_force_loss.mean(-1).sum(-1) / force_cnt).mean(), find_force
                    )
                    l1_force_loss = (l1_force_loss.sum(-1).sum(-1) / force_cnt).sum()
                else:
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_loss.mean().detach(), find_force
                    )
                    l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
                loss += (pref_f * l1_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            if mae:
                mae_f = torch.mean(torch.abs(diff_f))
                more_loss["mae_f"] = self.display_if_exist(mae_f.detach(), find_force)

        if self.has_v and "virial" in model_pred and "virial" in label:
            find_virial = label.get("find_virial", 0.0)
            pref_v = pref_v * find_virial
            diff_v = label["virial"] - model_pred["virial"].reshape(-1, 9)
            l2_virial_loss = torch.mean(torch.square(diff_v))
            if not self.inference:
                more_loss["l2_virial_loss"] = self.display_if_exist(
                    l2_virial_loss.detach(), find_virial
                )
            if not self.huber:
                loss += atom_norm * (pref_v * l2_virial_loss)
            else:
                l_huber_loss = custom_huber_loss(
                    model_pred["virial"], label["virial"], delta=self.huber_delta
                )
                loss += atom_norm * (pref_v * l_huber_loss)
            rmse_v = l2_virial_loss.sqrt() * atom_norm
            more_loss["rmse_v"] = self.display_if_exist(rmse_v.detach(), find_virial)
            if mae:
                mae_v = torch.mean(torch.abs(diff_v)) * atom_norm
                more_loss["mae_v"] = self.display_if_exist(mae_v.detach(), find_virial)
        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        #log.info(loss)
        #import sys
        #sys.exit()
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_e:
            label_requirement.append(
                DataRequirementItem(
                    "energy",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                )
            )
        if self.has_f:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_v:
            label_requirement.append(
                DataRequirementItem(
                    "virial",
                    ndof=9,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_ae:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_pf:
            label_requirement.append(
                DataRequirementItem(
                    "atom_pref",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                )
            )
        return label_requirement