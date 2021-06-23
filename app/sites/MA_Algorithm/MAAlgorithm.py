import numpy as np

def moving_average(x, w):
    result= np.convolve(x, np.ones(w), 'valid') / w
    z=x[0:w-1]
    result_cat=np.concatenate((z,result), axis=0)
    return result_cat
