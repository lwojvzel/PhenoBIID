"""Label-free history/forecast concatenation with masked trajectory pooling."""
import numpy as np

from run_history_multimodal_baselines import HISTORY_FEATURE_NAMES

INPUTS = ("history", "relative_valid", "previous_lai", "previous_lai_valid", "predicted_lai")
VARIANTS = ("history_only", "history_masks", "predicted_full", "predicted_mean", "previous_full", "previous_mean")
ENGINES = ("hgb", "lightgbm", "xgboost")


def masked_average(tokens, valid):
    """Pool [B,K,D] into [B,D]; an empty trajectory remains missing, not zero."""
    tokens, valid = np.asarray(tokens), np.asarray(valid, dtype=bool)
    if tokens.ndim != 3 or tokens.shape[:2] != valid.shape:
        raise ValueError("Expected tokens [B,K,D] and mask [B,K]")
    numerator = np.where(valid[..., None], tokens, 0.).sum(axis=1, dtype=np.float64)
    count = valid.sum(axis=1, keepdims=True)
    return np.divide(numerator, count, out=np.full_like(numerator, np.nan), where=count > 0).astype(np.float32)


def make_features(inputs, variant):
    if set(inputs) != set(INPUTS) or variant not in VARIANTS:
        raise ValueError("Only the declared past inputs and frozen LAI forecasts are allowed")
    history = np.asarray(inputs["history"], dtype=np.float32)
    n = len(history)
    if history.shape != (n, 15) or not np.isfinite(history).all():
        raise ValueError("Invalid historical yield features")
    if variant == "history_only":
        return history.copy(), list(HISTORY_FEATURE_NAMES)
    for key in INPUTS[1:]:
        if np.shape(inputs[key]) != (n, 12):
            raise ValueError(f"Invalid shape for {key}")
    active = np.asarray(inputs["relative_valid"], dtype=bool)
    observed = active & np.asarray(inputs["previous_lai_valid"], dtype=bool)
    count = active.sum(axis=1)
    metadata = np.stack((count / 12., observed.sum(axis=1) / np.maximum(count, 1)), axis=1).astype(np.float32)
    parts, names = [history, metadata], [*HISTORY_FEATURE_NAMES, "active_phase_fraction", "past_lai_observed_fraction"]
    if variant != "history_masks":
        source, representation = variant.split("_")
        valid = active if source == "predicted" else observed
        values = np.asarray(inputs[source + "_lai"], dtype=np.float32)
        if not np.isfinite(values[valid]).all():
            raise ValueError("Nonfinite values in a declared valid LAI slot")
        if representation == "full":
            remote = np.where(valid, values, np.nan)
            names.extend(f"{source}_lai_phase_{i+1:02d}" for i in range(12))
        else:
            remote = masked_average(values[..., None], valid)
            names.append(source + "_lai_masked_mean")
        parts.append(remote)
    return np.concatenate(parts, axis=1).astype(np.float32), names
