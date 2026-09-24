"""Unmodified definitions extracted from the registered source snapshot."""
import numpy as np

def trajectory_features(values, valid):
    """Keep every slot; summaries supplement, rather than replace, the trajectory."""
    valid = np.asarray(valid, bool)
    x = np.where(valid, values, 0).astype(np.float32)
    count = valid.sum(1)
    mean = x.sum(1)/np.maximum(count, 1)
    std = np.sqrt(np.where(valid, (x-mean[:, None])**2, 0).sum(1)/np.maximum(count, 1))
    maximum = np.where(count > 0, np.where(valid, x, -np.inf).max(1), 0)
    minimum = np.where(count > 0, np.where(valid, x, np.inf).min(1), 0)
    peak = np.where(count > 0, np.where(valid, x, -np.inf).argmax(1)/11., 0)
    return np.concatenate((x, np.stack((mean, std, maximum, minimum, x.sum(1), peak), 1)), 1).astype(np.float32)
