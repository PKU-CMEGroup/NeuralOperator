from .FCN import FCNet
from .train import FNN_train,count_params,HGkNN_train
from .newtrain import newHGkNN_train
from .Phytrain import PhyHGkNN_train
from .Phytrain_time import PhyHGkNN_train_time
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
