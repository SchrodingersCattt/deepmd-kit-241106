# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)

dtype = torch.float64

model_se_e2_a = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
    "atom_exclude_types": [1],
    "pair_exclude_types": [[1, 2]],
    "preset_out_bias": {
        "energy": [
            None,
            [1.0],
            [3.0],
        ]
    },
}


class TestGetModel(unittest.TestCase):
    def test_model_attr(self):
        model_params = copy.deepcopy(model_se_e2_a)
        self.model = get_model(model_params).to(env.DEVICE)
        atomic_model = self.model.atomic_model
        self.assertEqual(atomic_model.type_map, ["O", "H", "B"])
        self.assertEqual(
            atomic_model.preset_out_bias,
            {
                "energy": [
                    None,
                    torch.tensor([1.0], dtype=dtype, device=env.DEVICE),
                    torch.tensor([3.0], dtype=dtype, device=env.DEVICE),
                ]
            },
        )
        self.assertEqual(atomic_model.atom_exclude_types, [1])
        self.assertEqual(atomic_model.pair_exclude_types, [[1, 2]])

    def test_notset_model_attr(self):
        model_params = copy.deepcopy(model_se_e2_a)
        model_params.pop("atom_exclude_types")
        model_params.pop("pair_exclude_types")
        model_params.pop("preset_out_bias")
        self.model = get_model(model_params).to(env.DEVICE)
        atomic_model = self.model.atomic_model
        self.assertEqual(atomic_model.type_map, ["O", "H", "B"])
        self.assertEqual(atomic_model.preset_out_bias, None)
        self.assertEqual(atomic_model.atom_exclude_types, [])
        self.assertEqual(atomic_model.pair_exclude_types, [])

    def test_preset_wrong_len(self):
        model_params = copy.deepcopy(model_se_e2_a)
        model_params["preset_out_bias"] = {"energy": [None]}
        with self.assertRaises(ValueError):
            self.model = get_model(model_params).to(env.DEVICE)
