from .FCN import FCNet
from .train import FNN_train,count_params,myFNN_train
from .utils import (
    count_params,
    compute_1dFourier_bases,
    compute_2dFourier_bases,
    compute_1dWeights,
    compute_1dFourier_bases_arbitrary,
    compute_1dpca_bases,
    compute_2dpca_bases,
    compute_2dFourier_cbases,
    compute_H,
)
from .adam import Adam
from .losses import LpLoss
from .normalizer import UnitGaussianNormalizer
