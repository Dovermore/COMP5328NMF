"""
File: __init__.py
Author: Calvin Huang
Email: zhua9812@uni.sydney.edu.au
Github: https://github.com/dovermore
Description: Imports items from other submodules in this module for easier import statements
"""

from .misc import (load_data, load_data_AR, check_create_parent,
                   check_create_dir, get_current_time)
from .evaluation import (acc_score, benchmark, indent, make_grid_alg_kwargs,
                         nmi_score, rre_score, textwrap, assign_cluster_label)