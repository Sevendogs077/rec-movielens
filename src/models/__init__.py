from sklearn.linear_model import LogisticRegression

from .mf import MatrixFactorization
from .neumf import NeuMF
from .fm import FactorizationMachine
from .widedeep import WideDeep
# from .deepfm import DeepFM

all_models = {
    'mf': MatrixFactorization,
    'neumf': NeuMF,
    'fm': FactorizationMachine,
    'wide&deep': WideDeep,
}