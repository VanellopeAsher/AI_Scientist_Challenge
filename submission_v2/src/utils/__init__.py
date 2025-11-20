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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
