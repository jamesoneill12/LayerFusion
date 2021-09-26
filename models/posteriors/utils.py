from models.posteriors.hsoftmax import HierarchicalSoftmax
from models.posteriors.adasoftmax import AdaptiveSoftmax
from models.posteriors.target_sampling import TargetSampling
from models.posteriors.dsoftmax import DifferentiatedSoftmax
from models.posteriors.softmax_mixture import SoftmaxMixture
from models.posteriors.self_norm import selu


def choose_approx_softmax(approx):
    if approx == "hsoftmax":
        return HierarchicalSoftmax
    elif approx == "adasoftmax":
        return AdaptiveSoftmax
    elif approx == "target_sampling":
        return TargetSampling
    elif approx == "dsoftmax":
        return DifferentiatedSoftmax
    elif approx == "self_norm":
        return selu
    elif approx == "softmaxmixture":
        return SoftmaxMixture