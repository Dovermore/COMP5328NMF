from .nmf_l2 import NmfL2Estimator
from .nmf_hyper import NmfHyperEstimator

try:
    from .sknmf_benchmark import ModifiedNMF
except:
    pass
