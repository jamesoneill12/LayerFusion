from torch import nn


class HuberLoss(nn.Module):
    """
    The choice of delta is critical because it determines what you’re willing to consider as an outlier.
    Residuals larger than delta are minimized with L1 (which is less sensitive to large outliers),
    while residuals smaller than delta are minimized “appropriately” with L2.
    """
    def __init__(self, delta=0.1):
        super(HuberLoss, self).__init__()
        self.mse = nn.MSE()
        self.mae = nn.MAE()
        self.delta = delta

    def forward(self, y_pred, y_target, ):
        e = self.mse(y_pred, y_target)
        if e > self.delta:
            e = self.delta * self.mae(y_pred, y_target) + 1/2 * (self.delta ** 2)
        return e/y_target.shape[0]
