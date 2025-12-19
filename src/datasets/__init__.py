from .interaction import InteractionDataset
from .feature import FeatureDataset
# from .sequential import SequentialDataset

dataset_mapping = {
    # Interaction
    'mf': InteractionDataset,
    'gmf': InteractionDataset,
    'neumf': InteractionDataset,

    # Feature
    'fm': FeatureDataset,
    'wide_deep': FeatureDataset,
    'deepfm': FeatureDataset,

    # Sequential
    #'din': SequentialDataset,
    #'sasrec': SequentialDataset
}

def get_dataset(args):
    #return dataset_mapping[args.model_type](args)
    dataset_class = dataset_mapping.get(args.model_type)
    return dataset_class(args.data_path)