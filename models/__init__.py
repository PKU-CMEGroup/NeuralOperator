from .FCN import FCNet
from .train import FNN_train,count_params,HGkNN_train
from .Phytrain import PhyHGkNN_train
from .newPhytrain import newPhyHGkNN_train
from .GPtrain import GPtrain
from .GPtrain2 import GPtrain2
from .utils import (
    count_params,
    compute_1dFourier_bases,
    compute_2dFourier_bases,
    compute_1dWeights,
    compute_1dFourier_bases_arbitrary,
    compute_1dpca_bases,
    compute_2dpca_bases,
    compute_2dFourier_cbases,

)
from .adam import Adam
from .losses import LpLoss
from .normalizer import UnitGaussianNormalizer
