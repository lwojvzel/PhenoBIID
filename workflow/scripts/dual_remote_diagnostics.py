"""Explicit past-only inputs and separately labeled observation diagnostics."""
import numpy as np
import torch
from torch import nn

from dual_remote_state import YIELD_INPUTS
from run_input_matched_direct_yield import DirectGRU, DirectTransformer

DIRECT_CONDITIONS = ("none", "lai", "ndvi", "both", "observed_both")
OBSERVATION_CONDITIONS = ("observed_lai", "observed_ndvi", "observed_both")


def prepare_direct(a, condition):
    if condition not in DIRECT_CONDITIONS:
        raise ValueError(condition)
    b = dict(a)
    for product in ("lai", "ndvi"):
        enabled = condition in (product, "both", "observed_both")
        value = a[f"previous_{product}"]
        if condition == "observed_both":
            value = np.where(a[f"target_{product}_valid"] > 0, a[f"target_{product}"], value)
        b[f"state_{product}"] = value.copy() if enabled else np.zeros_like(value)
        if not enabled:
            b[f"previous_{product}_valid"] = np.zeros_like(value)
            if product == "ndvi":
                b["previous_ndvi_quality"] = np.zeros_like(value)
    return b


def prepare_observed(a, condition):
    if condition not in OBSERVATION_CONDITIONS:
        raise ValueError(condition)
    b = dict(a)
    for product in ("lai", "ndvi"):
        value = a[f"predicted_{product}"]
        if condition in (f"observed_{product}", "observed_both"):
            value = np.where(a[f"target_{product}_valid"] > 0, a[f"target_{product}"], value)
        b[f"state_{product}"] = value.copy()
    return b


def features(b):
    active = b["relative_valid"]
    sequence = torch.cat((b["weather"], b["state_lai"][..., None],
                          b["previous_lai_valid"][..., None], b["state_ndvi"][..., None],
                          b["previous_ndvi_valid"][..., None],
                          b["previous_ndvi_quality"][..., None], active[..., None]), -1)
    sequence = sequence * active[..., None]
    q = b["crop_coverage"]
    r = .1 + .9 * q / (q + .05)
    static = torch.cat((b["history"], b["context"], q[:, None], r[:, None], b["history_base"][:, None]), -1)
    return sequence, static, active


class DirectRemoteYield(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.network = {"gru": DirectGRU, "transformer": DirectTransformer}[architecture](19, 23, .1)

    def forward(self, b):
        if set(b) != set(YIELD_INPUTS):
            raise ValueError("Direct forward requires an explicitly prepared input packet, not labels")
        sequence, static, valid = features(b)
        if not valid.bool().any(1).all():
            # The shared Transformer needs one unmasked key for an empty sequence.
            valid = valid.clone()
            valid[:, 0] = torch.where(valid.any(1), valid[:, 0], torch.ones_like(valid[:, 0]))
        correction = self.network(sequence, static, valid)
        prediction = b["history_base"] + correction
        return dict(prediction=prediction, candidate=prediction, gate=torch.ones_like(prediction))
