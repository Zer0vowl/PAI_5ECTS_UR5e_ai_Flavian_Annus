import numpy as np
def rms(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))
