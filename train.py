import numpy as np

import pandas as pd
import pandas.core.tools.datetimes

import datetime
import dateutil.relativedelta

import tqdm
import lightgbm as lgbm

import sklearn.utils

import multiprocessing
import logging

import itertools


logging.basicConfig(level=logging.INFO, format='%(actime)s-%(name)s-%(level)s %(message)s')
logger = logging.getLogger(__name__)

def _guess_date_format(date):
    return pandas.core.tools.datetimes._guess_datetime_format(str(date))

def _convert_date_to_int(date):
    if not isinstance(date, datetime.date):
        raise ValueError('Input should be datetime object')
    return int(date.strftime('%Y%m%d'))

def _convert_date_to_string(date, format='%Y%m%d'):
    if not isinstance(date, datetime.date):
        raise ValueError('Input should be datetime object')
    return date.strftime(format)

def _parse_datetime(date):
    return datetime.datetime.strptime(str(date), _guess_date_format(date))

def get_rolling_date_slice(start, end, train, test, rolling=1, freq='months', expand=False):
    one_day = dateutil.relativedelta.relativedelta(days=1)
    time_delta = dateutil.relativedelta.relativedelta(**{freq:1})
    start_date = _parse_datetime(start)
    end_date = _parse_datetime(end)
    date_slices = []
    while start_date + train * time_delta < end_date:
        train_start = _convert_date_to_string(start_date)
        train_end = _convert_date_to_string(start_date + train * time_delta - one_day)
        test_start = _convert_date_to_string(start_date + train * time_delta)
        test_end = _convert_date_to_string(start_date + train * time_delta + test * time_delta)
        date_slices.append((slice(train_start, train_end), slice(test_start, test_end)))
        start_date += rolling * time_delta
    return date_slices

def _get_cv_data_generator(dataset, rolling_date_slice):
    if not isinstance(dataset, Dataset):
        raise ValueError('Input should be Dataset object')
    for date_slice in rolling_date_slice:
        train_X, train_y = dataset.slice_dataset(date_slice[0])
        test_X, test_y = dataset.slice_dataset(date_slice[1])
        if test_X.shape[0] > 0:
            yield train_X, train_y, test_X, test_y

def _slice_dataframe(df, date_slice):
    if df.index.nlevels < 2:
        raise Exception('Dataframe should contain date, ticker index')
    slicer = (date_slice,) + tuple([slice(None)] * (df.index.nlevels - 1))
    return df.loc[slicer, :].copy(deep=True)

def _bootstrap(*arrays, frac=1., replace=True):
    n_samples = min(array.shape[0] for array in arrays) * frac
    return sklearn.utils.resample(*arrays, n_samples=int(n_samples), replace=replace)

def train_model(task, model, parameters):
    if not task.valid:
        raise Exception('Invalid training task')
    train_X, train_y, test_X, test_y = task.train_data
    trained_model = model(**parameters).fit(train_X, train_y)
    return trained_model

def train_ensemble(task, model, parameters, num_ensemble, bagging_frac=.8):
    if not task.valid:
        raise Exception('Invalid training task')
    train_X, train_y, test_X, test_y = task.train_data
    trained_models = []
    for _ in range(num_ensemble):
        train_X_, train_y_ = _bootstrap(train_X, train_y, frac=bagging_frac)
        trained_model = model(**parameters).fit(train_X_, train_y_)
        trained_models.append(trained_model)
    ensemble = Ensemble(trained_models)
    return ensemble

def work_in_parallel(processes, func, args):
    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool, tqdm.tqdm(total=len(args), desc='parallel jobs') as pbar:
            update = lambda result: pbar.update()
            jobs = [pool.apply_async(func=func, args=(arg,), callback=update) for arg in args]
            results = [job.get() for job in jobs]
    else:
        results = [func(arg) for arg in tqdm.tqdm(args,'work in parallel')]
    return results



class Dataset:
    def __init__(self, df, label_cols):
        if df.index.nlevels < 2:
            raise Exception('Dataframe should contain date, ticker index')
        df_ = df.sort_index()
        self.label_cols = list(label_cols)
        self.feature_cols = list(df_.columns.difference(self.label_cols))
        self.features = df_[self.feature_cols]
        self.labels = df_[self.label_cols]

    def slice_dataset(self, date_slice):
        features = _slice_dataframe(self.features, date_slice)
        labels = _slice_dataframe(self.labels, date_slice)
        return features, labels

class _LargeDataStroage:
    def __init__(self):
        pass

def _memory_usage(*arrays):
    bytes = 0
    for array in arrays:
        if isinstance(array, pd.DataFrame):
            bytes += array.memory_usage(index=True, deep=True)
        if isinstance(array, np.ndarray):
            pass
    return bytes

def _generate_token(num_token):
    return

class LargeTask:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.namespace = multiprocessing.Manager().Namespace()
        self.valid = self._is_valid(train_X, train_y, test_X, test_y)
        self.desc = self._desc(train_X, train_y, test_X, test_y)
        self.queries = self._construct(train_X, train_y, test_X, test_y)

    def _construct(self, train_X, train_y, test_X, test_y):
        return

    def _is_valid(self, train_X, train_y, test_X, test_y):
        is_non_empty = all(df.shape[0] > 0 for df in [train_X, train_y, test_X, test_y])
        is_single_label = train_y.shape[1] == 1 and test_y.shape[1] == 1
        return is_non_empty & is_single_label

    @property
    def train_data(self):
        return self.train_X, self.train_y, self.test_X, self.test_y

    @property
    def train_X(self):
        return self._get('train_X')

    @property
    def train_y(self):
        return self._get('train_y')

    @property
    def test_X(self):
        return self._get('test_X')

    @property
    def test_y(self):
        return self._get('test_y')

    def _get(self, query_data):
        if query_data == 'train_X':
            return
        elif query_data == 'train_y':
            return
        elif query_data == 'train_y':
            return
        elif query_data == 'train_y':
            return
        else:
            raise TypeError('Unknown query data {}'.format(query_data))

    def _desc(self, train_X, train_y, test_X, test_y):
        num_train, num_test = train_X.shape[0], test_X.shape[0]
        if isinstance(train_X, pd.DataFrame):
            train_index = train_X.index
            test_index = test_X.index
            train_start, train_stop = train_index.min()[0], train_index.max()[0]
            test_start, test_stop = test_index.min()[0], test_index.max()[0]
            return 'train_start: {}, train_stop: {}, num_train: {}, test_start: {}, test_stop: {}, num_test: {}'.format(
                train_start, train_stop, num_train, test_start, test_stop, num_test)
class Task:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    @property
    def valid(self):
        is_non_empty = all(df.shape[0] > 0 for df in self.train_data)
        is_single_label = self.train_y.shape[1] == 1 and self.test_y.shape[1] == 1
        return is_non_empty & is_single_label

    @property
    def train_data(self):
        return self.train_X, self.train_y, self.test_X, self.test_y

    @property
    def desc(self):
        num_train, num_test = self.train_X.shape[0], self.test_X.shape[0]
        if isinstance(self.train_X, pd.DataFrame):
            train_index = self.train_X.index
            test_index = self.test_X.index
            train_start, train_stop = train_index.min()[0], train_index.max()[0]
            test_start, test_stop = test_index.min()[0], test_index.max()[0]
            return 'train_start: {}, train_stop: {}, num_train: {}, test_start: {}, test_stop: {}, num_test: {}'.format(
                train_start, train_stop, num_train, test_start, test_stop, num_test)

class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X, weight=None):
        predictions = np.array([model.predict(X) for model in self.models])
        if weight == 0:
            return np.mean(predictions, axis=0)
        else:
            weight_ = np.array(weight).reshape((-1,1))
            if len(predictions) != len(weight_):
                raise ValueError('Incompatible shape of ensemble weight')
            return np.dot(predictions, weight_)

def create_cv_tasks(dataset, start, end, train, test, rolling=1, freq='months', expand=False):
    tasks = []
    cv_date_slices = get_rolling_date_slice(start, end, train, test, rolling, freq, expand)
    for train_X, train_y, test_X, test_y in tqdm.tqdm(_get_cv_data_generator(dataset, cv_date_slices), desc='Create CV tasks'):
        task = Task(train_X, train_y, test_X, test_y)
        tasks.append(task)
    return tasks

def create_custom_cv_tasks(dataset, date_slices):
    tasks = []
    for train_X, train_y, test_X, test_y in tqdm.tqdm(_get_cv_data_generator(dataset, date_slices),
                                                      desc='Create CV tasks'):
        task = Task(train_X, train_y, test_X, test_y)
        tasks.append(task)
    return tasks
    pass

class TrainEngine:
    def __init__(self, tasks, processes):
        self.processes = processes
        self.tasks = tasks

    @property
    def desc(self):
        return '\n'.join([task.desc for task in self.tasks])

    def train(self, model, parameter, num_ensemble=1, bagging_frac=.8):
        if self.tasks == 0:
            raise Exception('Empty task list')
        if num_ensemble > 1:
            args = [(task, model, parameter, num_ensemble, bagging_frac) for task in self.tasks]
            results = work_in_parallel(self.processes, train_ensemble, args)
        else:
            args = [(task, model, parameter) for task in self.tasks]
            results = work_in_parallel(self.processes, train_model, args)
        return results



def f(a):
    return a*a

if __name__ == '__main__':

    # df = pd.read_hdf(r'/home/wangmengnan/Downloads/magic/all_factors_889.h5')
    # df = df.set_index(['date','code'])
    # # # print(_guess_date_format('2018-01-01'))
    # #
    # dataset = Dataset(df,'y')
    # train_engine = TrainEngine(dataset, 2)
    # train_engine.create_cv_tasks('2018-01-01','2018-06-01',3,1, expand=True)
    # print(work_in_parallel(6, f, range(10)))
    # train_X, train_y = dataset.slice_dataset(slice('20180101','20180301'))
    # test_X, test_y = dataset.slice_dataset(slice('20180401','20180501'))
    # task = Task(train_X, train_y, test_X, test_y)
    # a = np.ones((10,2))
    # b = np.ones((10,2))
    # c,d = _bootstrap(a, b, frac=.8)
    pass
