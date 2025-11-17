from .api_utils import *
from .llm_utils import *
from .file_utils import *
from .prompt_utils import *
from .vllm import *
from .excel_utils import *
from .plot_utils import *
from .paper_manager_utils import *

import random
import numpy as np

# Optional torch import (not required for API service)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
