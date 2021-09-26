from .hsoftmax import HierarchicalSoftmax
from .dsoftmax import DifferentiableSoftmax
from models.loss.nce import NCELoss


def get_posterior(approxiamte):

    if approxiamte == "nce":
        return NCELoss
    elif approxiamte == "hsoftmax":
        return HierarchicalSoftmax
    elif approxiamte == "dsoftmax":
        return DifferentiableSoftmax