import pandas as pd
import numpy as np
from jqdatasdk import *
from jqdatasdk import finance
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
from numba import njit
# auth('', '')
# a_shares_list = get_index_stocks('000001.XSHG')
# get_price(security, start_date=None, end_date=None, frequency='daily', fields=None, skip_paused=False, fq='pre', count=None)



# daily = get_price(a_shares_list, start_date='2018-01-01', end_date='2019-06-01', frequency='daily',
#                   fields=['open', 'close', 'low', 'high', 'volume', 'money', 'factor', 'high_limit','low_limit',
#                           'avg', 'pre_close', 'paused'], skip_paused=False, fq='post', count=None).to_frame()
# pass
# numpy func
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

def _gen_matrix(val, a):
    return np.full(a.shape, val)

def _wma(a):
    length = a.shape[0]
    weight = (np.arange(length)[::-1])* 0.9
    return np.dot(weight, a)

def _decaylinear(a):
    length = a.shape[0]
    weight = np.arange(length)[::-1]+1
    weight = weight/weight.sum()
    return np.dot(weight,a)
#todo
def _regresi(x, y):
    x_demean = x - x.mean(axis=0)
    y_demean = y - y.mean(axis=0)
    beta1 = (x_demean * y_demean).sum(axis=0)/(x_demean**2).sum(axis=0)
    beta0 = y.mean(axis=0) - beta1 * x.mean(axis=0)
    y_hat = beta0 + beta1 * x
    e = y - y_hat
    return (e**2).mean(axis=0)
#todo
def _regbeta(x, y):
    x_demean = x - x.mean(axis=0)
    y_demean = y - y.mean(axis=0)
    beta1 = (x_demean * y_demean).sum(axis=0)/(x_demean**2).sum(axis=0)
    return beta1
#todo
def _regbeta_seq(x):
    # x = np.nan_to_num(x)
    length = x.shape[0]
    seq = SEQUENCE(length).reshape((-1,1))
    x_demean = x - x.mean(axis=0)
    y_demean = np.repeat(seq - seq.mean(axis=0), x.shape[1],axis=1)
    beta1 = (x_demean * y_demean).sum(axis=0)/(x_demean**2).sum(axis=0)
    return beta1

# @njit
def _sma(a, n, m):
    length = a.shape[0]
    results = [np.nan_to_num(a[0])]
    for i in range(length - 1):
        results.append(np.nan_to_num(a[i]) * m / n + results[i] * (n - m) / n)
    return results

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

def MAX(X, Y):
    return COND(X>=Y, X, Y)

def MIN(X, Y):
    return COND(X<=Y, X, Y)

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

def COND(C, X, Y):
    C_array = C.values
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = _gen_matrix(X, C_array)
    if isinstance(Y, pd.DataFrame):
        Y_array = Y.values
    else:
        Y_array = _gen_matrix(Y, C_array)
    results = np.where(C_array, X_array, Y_array)
    return pd.DataFrame(results, index=C.index, columns=C.columns)

def WMA(X, d):
    X_array = X.values
    results = _backward_rolling(d, _wma, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def DECAYLINEAR(X, d):
    X_array = X.values
    results = _backward_rolling(d, _decaylinear, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def SMA(X, n, m):
    if n <= m:
        raise ValueError('n must be greater than m')
    X_array = X.values
    results = _sma(X_array, n, m)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def LOG(X):
    return np.log(X)

def ABS(X):
    return abs(X)

def SIGN(X):
    return np.sign(X)
#todo
def REGBETA(X, Y, d):
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.repeat(X, Y.shape[1])
    if isinstance(Y, pd.DataFrame):
        Y_array = Y.values
    else:
        Y_array = np.repeat(Y, X.shape[1])
    results = _backward_rolling(d, _regbeta, X_array, Y_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def REGBETA_SEQ(X, d):
    X_array = X.values
    results = _backward_rolling(d, _regbeta_seq, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)

def FAMA3_RESI(X, MKT, SMB, HML):
    return

def FAMA5_RESI(X, MKT, SMB, HML):
    return
#todo
def REGRESI(X, Y, d):
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.repeat(X, Y.shape[1])
    if isinstance(Y, pd.DataFrame):
        Y_array = Y.values
    else:
        Y_array = np.repeat(Y, X.shape[1])
    results = _backward_rolling(d, _regresi, X_array, Y_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)
#todo
def PROD(X, d):
    X_array = X.values
    results = _backward_rolling(d, _prod, X_array)
    return pd.DataFrame(results, index=X.index, columns=X.columns)
#todo
def INDUSTYNEUTRALIZE(X):
    coids = None
    return
def SEQUENCE(n):
    return np.arange(6) + 1

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

# INGREDIENT
OPEN = daily.open.unstack()[coids].fillna(method='ffill')
HIGH = daily.high.unstack()[coids].fillna(method='ffill')
LOW = daily.low.unstack()[coids].fillna(method='ffill')
CLOSE = daily.close.unstack()[coids].fillna(method='ffill')
PRECLOSE = daily.pre_close.unstack()[coids].fillna(method='ffill')
VWAP = daily.avg.unstack()[coids].fillna(method='ffill')
VOLUME = daily.volume.unstack()[coids].fillna(method='ffill')
ADV = daily.money.unstack()[coids].fillna(method='ffill')
BANCHMARKINDEXOPEN = None
BANCHMARKINDEXCLOSE = None

# basic
RET = CLOSE/PRECLOSE - 1
DTM = COND(OPEN<=DELAY(OPEN,1), 0, MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
DBM = COND(OPEN>=DELAY(OPEN,1), 0, MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
TR = MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
HD = HIGH - DELAY(HIGH, 1)
LD = DELAY(LOW, 1) - LOW
HML = None
SMB = None
MKE = None


alpha191 = pd.read_hdf(r'/home/wangmengnan/Downloads/alpha191.h5')
alpha = 'alpha191_alpha_008'
jq_alpha = alpha191[['date','code',alpha]].set_index(['date','code'])[alpha].unstack()
#TODO protect div

# WMA(OPEN,5)
# corr = CORR(OPEN,VWAP,5)
# ts_rank = TSRANK(OPEN,5)
# alpha001 = (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6))
# alpha002 = -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/((HIGH-LOW)),1)
# alpha003 = SUM(COND(CLOSE>DELAY(CLOSE,1),0,CLOSE-COND(CLOSE>DELAY(CLOSE,1),MIN(LOW,DELAY(CLOSE,1)),MAX(HIGH,DELAY(CLOSE,1)))),6)
# alpha004 = COND((((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2)),(-1*1),COND(((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8))),1,COND(((1<(VOLUME/MEAN(VOLUME,20)))|((VOLUME/MEAN(VOLUME,20))==1)),1,-1)))
# alpha005 = (-1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))
# alpha006 = (RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))),4)))*-1)
# alpha007 = ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))
# alpha008 = RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4)-1)
# alpha009 = SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2) * 10e8
# alpha010 = RANK(COND(RET<0,STD(RET,20),CLOSE)**2) #TODO missing MAX
# alpha011 = SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
# alpha012 = (RANK((OPEN-(SUM(VWAP,10)/10))))*(-1*(RANK(ABS((CLOSE-VWAP)))))
# alpha013 = (((HIGH*LOW)**0.5)-VWAP)
# alpha014 = CLOSE-DELAY(CLOSE,5)
# alpha015 = OPEN/DELAY(CLOSE,1)-1
# alpha016 = (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))
# alpha017 = RANK((VWAP-MAX(VWAP,15)))**DELTA(CLOSE,5)
# alpha018 = CLOSE/DELAY(CLOSE,5)
# alpha019 = COND(CLOSE<DELAY(CLOSE,5),(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5),COND(CLOSE==DELAY(CLOSE,5),0,(CLOSE-DELAY(CLOSE,5))/CLOSE))
# alpha020 = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
# alpha021 = REGBETA_SEQ(MEAN(CLOSE,6),6)
# alpha022 = SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
# alpha023 = SMA(COND(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)/(SMA(COND(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)+SMA(COND(CLOSE<=DELAY(CLOSE,1),STD(CLOSE,20),0),20,1))*100
# alpha024 = SMA(CLOSE-DELAY(CLOSE,5),5,1)
# alpha025 = ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))*(1+RANK(SUM(RET,250)))) #TODO 250 is too long
# alpha026 = ((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230)))) #TODO 230 is too long
# alpha027 = WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
# alpha028 = 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9)+0.001)*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/( MAX(HIGH,9)-TSMAX(LOW,9)+0.001)*100,3,1),3,1)
# alpha029 = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
# alpha030 = WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML,60))**2,20) #TODO MKT,SMB,HML
alpha031 =
alpha032 =
alpha033 =
alpha034 =
alpha035 =
alpha036 =
alpha037 =
alpha038 =
alpha039 =
alpha040 =
alpha041 =
alpha042 =
alpha043 =
alpha044 =
alpha045 =
alpha046 =
alpha047 =
alpha048 =
alpha049 =
alpha050 =
alpha051 =
alpha052 =
alpha053 =
alpha054 =
alpha055 =
alpha056 =
alpha057 =
alpha058 =
alpha059 =
alpha060 =
alpha061 =
alpha062 =
alpha063 =
alpha064 =
alpha065 =
alpha066 =
alpha067 =
alpha068 =
alpha069 =
alpha070 =
alpha071 =
alpha072 =
alpha073 =
alpha074 =
alpha075 =
alpha076 =
alpha077 =
alpha078 =
alpha079 =
alpha080 =
alpha081 =
alpha082 =
alpha083 =
alpha084 =
alpha085 =
alpha086 =
alpha087 =
alpha088 =
alpha089 =
alpha090 =
alpha091 =
alpha092 =
alpha093 =
alpha094 =
alpha095 =
alpha096 =
alpha097 =
alpha098 =
alpha099 =
alpha100 =
alpha101 =
alpha102 =
alpha103 =
alpha104 =
alpha105 =
alpha106 =
alpha107 =
alpha108 =
alpha109 =
alpha110 =
# sma = SMA(OPEN,5,2)
# a = np.ones(10).astype(bool)
pass

# logout()
