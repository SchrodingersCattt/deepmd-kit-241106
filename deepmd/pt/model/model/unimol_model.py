# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import torch.nn as nn
from typing import Dict, Any, List, Callable
from .unimol_core_model import UniMolCoreModel

log = logging.getLogger(__name__)

class UniMolModel(UniMolCoreModel):
    def __init__(self, model_params):
        log.info(model_params)
        super().__init__()