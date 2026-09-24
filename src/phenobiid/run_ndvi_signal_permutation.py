"""Unmodified definitions extracted from the registered source snapshot."""


def compose(config, base, component):
    norm = config['normalization']
    center = norm.get('center', norm.get('residual_mean'))
    scale = norm.get('scale', norm.get('residual_std'))
    if config['head'] == 'lightgbm':
        return base+scale*component+center
    return base+center+scale*component
