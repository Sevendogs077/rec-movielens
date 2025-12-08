from .mf import MatrixFactorization
from .gmf import GeneralizedMF
from .ncf import NeuralCF
# from .fm import FactorizationMachine
# from .deepfm import DeepFM

all_models = {
    'mf': MatrixFactorization,
    'gmf': GeneralizedMF,
    'ncf': NeuralCF,
    # 'fm': FactorizationMachine
}