import numpy as np

def argmax(q_values):
    mask = q_values == q_values.max(axis=1, keepdims = True)
    r_noise = 1e-6 * np.random.random(q_values.shape)
    return np.argmax(r_noise*mask, axis = 1)
