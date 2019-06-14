import pandas as pd
import numpy as np
from jqdatasdk import *
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
from numba import njit
# auth('', '')
# a_shares_list = get_index_stocks('000001.XSHG')

# daily = get_price(a_shares_list, start_date='2018-01-01', end_date='2019-06-01', frequency='daily',
#                   fields=['open', 'close', 'low', 'high', 'volume', 'money', 'factor', 'high_limit','low_limit',
#                           'avg', 'pre_close', 'paused'], skip_paused=False, fq='pre', count=None).to_frame()

# numpy func
@njit
def _corr(a,b):
    a_demean = a - a.mean(axis=0)
    b_demean = b - b.mean(axis=0)
    a_std = a.std(axis=0)
    b_std = b.std(axis=0)
    pearsonr = (a_demean * b_demean).mean(axis=0)/a_std/b_std
    return pearsonr

def _cov(a,b):
    a_demean = a - a.mean(axis=0)
    b_demean = b - b.mean(axis=0)
    covariance = (a_demean * b_demean).mean(axis=0)
    return covariance
#todo
def _wma(a):
    return

def _tsrank(a):
    order = np.argsort(a, axis=0)
    ranks = np.argsort(order, axis=0)
    return ranks[-1]

def _argmax(a):
    length = a.shape[1]
    return length - a.argmax(axis=0) - 1

def _argmin(a):
    length = a.shape[1]
    return length - a.argmin(axis=0) - 1

def _prod(a):
    return np.prod(a, axis=0)

def _sumif(c, x):
    return (c.astype(int) * x).sum(axis=0)

# rolling func
def _split(cube):
    length = cube.shape[0]
    return [cube[i,:,:] for i in range(length)]

def _return_nan(window, func, *arrays):
    if arrays[0].shape[0]<window:
        return np.full(arrays[0].shape[1], np.nan)
    return func(*arrays)

def _backward_rolling(window, func, *arrays):
    length = arrays[0].shape[0]
    cube = np.array(arrays)
    check_low_index = lambda a: 0 if a < 0 else a
    results = Parallel(prefer='threads',n_jobs=-1)(
        delayed(_return_nan)(window, func, *_split(cube[:,check_low_index(i + 1 - window): i + 1, :])) for i in
        range(length))
    return results

# def _forward_rolling(window, func, *arrays):
#     length = arrays[0].shape[0]
#     cube = np.array(arrays)
#     check_high_index = lambda a: 0 if a < 0 else a
#
#     results = Parallel(prefer='threads')(
#         delayed(return_nan_wrapper)(func, window, *np.split(cube[check_low_index(i - window):i + 1,:,:], len(arrays), axis=-1)) for i in
#         range(length))
#     return results

# alpha func
def DELTA(X, d):
    return X.diff(d)

def DELAY(X, d):
    return X.shift(d)

def RANK(X):
    return X.rank(axis=1)

def MEAN(X, d):
    return X.rolling(d).mean()

def STD(X, d):
    return X.rolling(d).std()

def CORR(X, Y, d):
    X_array = X.values
    Y_array = Y.values
    results = _backward_rolling(d, _corr, X_array, Y_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns).fillna(method='ffill')

def COV(X, Y, d):
    X_array = X.values
    Y_array = Y.values
    results = _backward_rolling(d, _cov, X_array, Y_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def COUNT(C, d):
    return C.astype(bool).rolling(d).sum()

def TSRANK(X, d):
    X_array = X.values
    results = _backward_rolling(d, _tsrank, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def TSMAX(X, d):
    return X.rolling(d).max()

def ARGMAX(X, d):
    X_array = X.values
    results = _backward_rolling(d, _argmax, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def ARGMIN(X, d):
    X_array = X.values
    results = _backward_rolling(d, _argmin, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def MAX(X, d):
    return X.rolling(d).max()

def TSMIN(X, d):
    return X.rolling(d).min()



def SUM(X, d):
    return X.rolling(d).sum()

def SUMIF(C, X, d):
    X_array = X.values
    C_array = C.values
    if C_array.dtype != bool:
        raise TypeError('condition must be boolean value')
    results = _backward_rolling(d, _sumif, C_array, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

#todo
def _weight(a, w):
    return

#todo
def WMA(X, d):
    X_array = X.values
    results = _backward_rolling(d, _wma, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

@njit
def _sma(a, n, m):
    length = a.shape[0]
    results = [a[0]]
    for i in range(length - 1):
        results.append(a[i] * m / n + results[i] * (n - m) / n)
    return results

def SMA(X, n, m):
    if n <= m:
        raise ValueError('n must be greater than m')
    X_array = X.values
    return pd.DataFrame(_sma(X_array, n, m), index=X.index, columns=X.columns)

def LOG(X):
    return np.log(X)

def ABS(X):
    return abs(X)

def SIGN(X):
    return np.sign(X)

def _regbeta(x, y):
    x_demean = x - x.mean(axis=0)
    y_demean = y - y.mean(axis=0)
    beta1 = (x_demean * y_demean).sum(axis=0)/(x_demean**2).sum(axis=0)
    return beta1

def REGBETA(X, Y, d):
    X_array = X.values
    Y_array = Y.values
    results = _backward_rolling(d, _regbeta, X_array, Y_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)
#todo
def _regresi(x, y):
    x_demean = x - x.mean(axis=0)
    y_demean = y - y.mean(axis=0)
    beta1 = (x_demean * y_demean).sum(axis=0)/(x_demean**2).sum(axis=0)
    beta0 = y.mean(axis=0) - beta1 * x.mean(axis=0)
    y_hat = beta0 + beta1 * x
    e = y - y_hat
    return (e**2).mean(axis=0)

def REGRESI(X, Y, d):
    X_array = X.values
    Y_array = Y.values
    results = _backward_rolling(d, _regresi, X_array, Y_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

#todo
def PROD(X, d):
    X_array = X.values
    results = _backward_rolling(d, _prod, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def INDUSTYNEUTRALIZE(X):
    coids = None

    return

MIN = TSMIN
HIGHDAY = ARGMAX
LOWDAY = ARGMIN
SUMAC = SUM
# a = np.array([1,2,3,4,5]).reshape((-1,1))
# b = np.array([1,2,3,4,5][::-1]).reshape((-1,1))
# _corr(a,b,5)
import jqdatasdk

daily = pd.read_hdf(r'/home/wangmengnan/Downloads/daily.h5')
# daily.index.names = ['date','coid']
# daily['ret'] = daily.close/daily.pre_close - 1
coids = daily.index.levels[-1].unique()


OPEN = daily.open.unstack()[coids].fillna(method='ffill')
HIGH = daily.high.unstack()[coids].fillna(method='ffill')
LOW = daily.low.unstack()[coids].fillna(method='ffill')
CLOSE = daily.close.unstack()[coids].fillna(method='ffill')
PRECLOSE = daily.pre_close.unstack()[coids].fillna(method='ffill')
VWAP = daily.avg.unstack()[coids].fillna(method='ffill')
VOLUME = daily.volume.unstack()[coids].fillna(method='ffill')
ADV = daily.money.unstack()[coids].fillna(method='ffill')

RET = CLOSE/PRECLOSE - 1
DTM = None
DBM = None
TR = None
HD = HIGH - DELAY(HIGH, 1)
LD = DELAY(LOW, 1) - LOW
HML = None
SMB = None
MKE = None
BANCHMARKINDEXOPEN = None
BANCHMARKINDEXCLOSE = None

alpha191 = pd.read_hdf(r'/home/wangmengnan/Downloads/alpha191.h5')
alpha = 'alpha191_alpha_008'
jq_alpha = alpha191[['date','code',alpha]].set_index(['date','code'])[alpha].unstack()
# corr = CORR(OPEN,VWAP,5)
# ts_rank = TSRANK(OPEN,5)
# alpha001 = (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6))
# alpha002 = -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/((HIGH-LOW)),1)

alpha003 = 'SUM((CLOSE>DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)'
# alpha005 = (-1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))
# alpha006 = (RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))),4)))*-1)
# alpha007 = ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))
# alpha008 = RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4)-1)
sma = SMA(OPEN,5,2)
# a = np.ones(10).astype(bool)
pass

# logout()
