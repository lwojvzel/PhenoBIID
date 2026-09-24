"""Single-factor masks for the already registered direct seasonal interface."""
import numpy as np

CONDITIONS = ('no_remote','no_weather','no_shared_quality')


def ablate(inputs,condition):
    if condition not in CONDITIONS:
        raise ValueError('Unregistered direct input condition')
    sequence,static = np.array(inputs['sequence'],copy=True),np.array(inputs['static'],copy=True)
    products = (sequence.shape[-1]-38)//7
    if products not in (1,2) or sequence.shape[1:] != (12,38+7*products) or static.shape != (len(sequence),21+24*products):
        raise ValueError('Unexpected direct feature layout')
    if condition == 'no_remote':
        sequence[...,5:11] = 0
        sequence[...,38:] = 0
        static[:,21:] = 0
    elif condition == 'no_weather':
        sequence[...,11:24] = 0
        sequence[...,24:37] = 0
    else:
        sequence[...,5:11] = 0
    return dict(inputs,sequence=sequence,static=static)
