from .mf import MatrixFactorization
from .gmf import GeneralizedMF
from .neumf import NeuMF
# from .fm import FactorizationMachine
# from .deepfm import DeepFM

all_models = {
    'mf': MatrixFactorization,
    'gmf': GeneralizedMF,
    'neumf': NeuMF,
    # 'fm': FactorizationMachine
}