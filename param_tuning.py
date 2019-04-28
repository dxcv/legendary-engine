import train
import pandas as pd
import itertools

def create_param_grid(parameters, dict_format=True,):
    param_grid = itertools.product(*parameters.values())
    if dict_format:
        return [dict(zip(parameters,param)) for param in param_grid]
    return param_grid

def grid_search():
    return



if __name__ == '__main__':
    df = pd.read_hdf(r'/home/wangmengnan/Downloads/magic/all_factors_889.h5')
    df = df.set_index(['date','code'])

    c = create_param_grid({'a': [1, 2], 'b': [3, 4]})