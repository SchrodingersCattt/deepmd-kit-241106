# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

from typing import (
    Dict,
)

import torch


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, PropertyFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: Dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        """Apply the stat to each atomic output.
        The developer may override the method to define how the bias is applied
        to the atomic output of the model.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc
        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        if self.fitting_net.bias:
            for kk in self.bias_keys:
                # nf x nloc x odims, out_bias: ntypes x odims
                ret[kk] = ret[kk] + out_bias[kk][atype]
        return ret

    def get_intensive(self) -> bool:
        """Get whether the property is intensive."""
        return self.atomic_output_def()["property"].get_intensive()
