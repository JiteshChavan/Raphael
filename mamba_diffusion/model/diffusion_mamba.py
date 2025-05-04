import math
from functools import partial
from typing import Optional

# makes it easy to save and load models by extending this class
from huggingface_hub import PyTorchModelHubMixin

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import CondMamba, Mamba
from pe.cpe import AdaInPosCNN