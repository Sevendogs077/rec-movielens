
from .funksvd import FunkSVD
from .neumf import NeuMF
from .lr import LogisticRegression
from .fm import FactorizationMachine
from .widedeep import WideDeep
# from .deepfm import DeepFM

all_models = {
    'funksvd': FunkSVD,
    'neumf': NeuMF,
    'lr': LogisticRegression,
    'fm': FactorizationMachine,
    'wide&deep': WideDeep,
}