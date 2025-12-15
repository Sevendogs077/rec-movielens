import torch.nn as nn

class BaseModel(nn.Module):
    REQUIRED_FEATURES = None # Subclass need to cover it

    def __init__(self, feature_dims):
        super().__init__()
        self._check_feature_equality(feature_dims)

    def _check_feature_equality(self, feature_dims):
        # Check if <feature_dims> equals to <REQUIRED_FEATURES>

        input_keys = set(feature_dims.keys())

        if isinstance(self.REQUIRED_FEATURES, list):
            required_keys = set(self.REQUIRED_FEATURES)

            missing = required_keys - input_keys
            if missing:
                raise ValueError(f"[{self.__class__.__name__}] Missing required features: {missing}")

            unexpected = input_keys - required_keys
            if unexpected:
                raise ValueError(f"[{self.__class__.__name__}] Received unexpected features: {unexpected}. "
                                 f"Model only requires: {self.REQUIRED_FEATURES}")

        elif self.REQUIRED_FEATURES == '__all__':
            if not input_keys:
                raise ValueError(f"[{self.__class__.__name__}] Input feature_dims cannot be empty!")

        else:
            raise TypeError(
                f"[{self.__class__.__name__}] Invalid REQUIRED_FEATURES definition. Must be list or '__all__'.")