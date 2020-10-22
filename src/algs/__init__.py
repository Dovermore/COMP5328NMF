"""
File: __init__.py
Author: Calvin Huang
Email: zhua9812@uni.sydney.edu.au
Github: https://github.com/dovermore
Description: Imports items from other submodules in this module for easier import statements
"""

from .nmf_l2 import NmfL2Estimator
from .nmf_l1 import NmfL1Estimator
from .nmf_hyper import NmfHyperEstimator

try:
    from .sknmf_benchmark import ModifiedNMF
except:
    pass
