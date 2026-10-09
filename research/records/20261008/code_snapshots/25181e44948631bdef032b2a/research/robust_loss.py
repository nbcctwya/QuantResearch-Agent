"""Optional Huber regression term; default training remains exactly unchanged."""
import math

from torch.nn import functional as F


def validate_delta(delta):
    if delta is None:
        return None
    if isinstance(delta,bool) or not isinstance(delta,(float,int)) or not math.isfinite(delta) or delta<=0:
        raise ValueError("Huber delta must be positive and finite, or None for the original squared loss")
    return float(delta)


def training_huber_delta(config):
    delta=validate_delta(config.get("huber_delta"))
    if delta is not None and (config["family"]!="residual" or config.get("objective","mse") not in ("mse","mixed")):
        raise ValueError("Huber experiment requires the scalar residual network and MSE or mixed objective")
    return delta


def robust_regression_loss(predictions,targets,delta=None):
    delta=validate_delta(delta)
    if delta is None:
        return F.mse_loss(predictions,targets)
    return 2*F.huber_loss(predictions,targets,delta=delta)


def robust_loss_spec(config):
    delta=training_huber_delta(config)
    if delta is None:
        return None
    return {"version":1,"huber_delta":delta,"objective":config.get("objective","mse"),
        "regression":"error squared for abs(error)<=delta; 2*delta*abs(error)-delta squared otherwise; mean over one forecast date",
        "torch_definition":"2 * torch.nn.functional.huber_loss; same quadratic coefficient as original MSE",
        "mixed_weights":{"regression":.5,"correlation":.5} if config.get("objective","mse")=="mixed" else None,
        "threshold_units":"existing training target units; no validation/test fitting",
        "default_disabled":"missing or None uses original MSE exactly",
        "predictions_and_evaluation":"original scalar model outputs and baseline ranking/backtest; no label or position changes"}
