import numpy as np
import time

from .fast_methods import _compute_val_metrics
from .fast_methods import _initialization
from .fast_methods import _run_epoch
from .fast_methods import _shuffle
from .utils import timer


class SVD():
    """Реализация алгоритма Simon Funk SVD 
    Attributes:
        lr (float): скорость обучения
        reg (float): коэффициент регуляризации
        n_epochs (int): количество итераций
        n_factors (int): количество признаков
        global_mean (float): среднее по всем оценкам
        pu (numpy array): матрица признаков пользователей
        qi (numpy array): матрица признаков элементов
        bu (numpy array): вектор смециений пользователей
        bi (numpy array): вектор смециений элемнетов
        early_stopping (boolean): стоит ли прекратить обучение на основе вычисления ошибок на валидационной выборке
        shuffle (boolean): стоит ли перемешивать данные перед каждой эпохой.
    """

    def __init__(self, learning_rate=.005, regularization=0.02, n_epochs=20,
                 n_factors=100, min_rating=1, max_rating=5):

        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating

    def _preprocess_data(self, X, train=True):
        """Сопоставляет идентификаторы пользователей и элементов с индексами и возвращает массив NumPy.
        Args:
            X (pandas DataFrame): dataset
            train (boolean): флаг, который определяет является ли Х обучающим набором
        Returns:
            X (numpy array): сопоставленный набор данных
        """
        X = X.copy()

        if train:
            u_ids = X['u_id'].unique().tolist()
            i_ids = X['i_id'].unique().tolist()

            self.user_dict = dict(zip(u_ids, [i for i in range(len(u_ids))]))
            self.item_dict = dict(zip(i_ids, [i for i in range(len(i_ids))]))

        X['u_id'] = X['u_id'].map(self.user_dict)
        X['i_id'] = X['i_id'].map(self.item_dict)

        # Tag unknown users/items with -1 (when val)
        X.fillna(-1, inplace=True)

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        X = X[['u_id', 'i_id', 'rating']].values

        return X

    def _sgd(self, X, X_val):
        """Алгоритм SGD
        Args:
            X (numpy array): обучающий набор, первый столбец должен содержать индексы пользователей, 
		второго - индексы элементов, и третьего - рейтинга.
            X_val (numpy array or `None`): валидационный набор данных
        """
        n_user = len(np.unique(X[:, 0]))
        n_item = len(np.unique(X[:, 1]))

        pu, qi, bu, bi = _initialization(n_user, n_item, self.n_factors)

        if self.early_stopping:
            list_val_rmse = [float('inf')]

        # SGD
        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)

            pu, qi, bu, bi = _run_epoch(X, pu, qi, bu, bi, self.global_mean,
                                        self.n_factors, self.lr, self.reg)

            
            if X_val is not None:
                val_metrics = _compute_val_metrics(X_val, pu, qi, bu, bi,
                                                       self.global_mean,
                                                       self.n_factors)
                
                val_loss, val_rmse, val_mae = val_metrics
                self._on_epoch_end(start, val_loss, val_rmse, val_mae)
                if self.early_stopping:
                    
                    list_val_rmse.append(val_rmse)

                    if self._early_stopping(list_val_rmse):
                        break

            else:
                self._on_epoch_end(start)

        self.pu = pu
        self.qi = qi
        self.bu = bu
        self.bi = bi

    @timer(text='\nTraining took ')
    def fit(self, X, X_val=None, early_stopping=False, shuffle=False):
        """Настройка параметров модели
        Args:
            X (pandas DataFrame): обучающий набор, должен иметь столбец `u_id` для идентификатора пользователя,
                столбец «i_id» для идентификатора элемента и «rating».
            X_val (pandas DataFrame, defaults to `None`): валидационный набор данных
            early_stopping (boolean): стоит ли прекратить обучение на основе вычисления ошибок на валидационной выборке
	    shuffle (boolean): стоит ли перемешивать данные перед каждой эпохой.
        Returns:
            self (SVD object): обученная модель
        """
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        print('Preprocessing data...\n')
        X = self._preprocess_data(X)

        if X_val is not None:
            X_val = self._preprocess_data(X_val, train=False)

        self.global_mean = np.mean(X[:, 2])
        self._sgd(X, X_val)

        return self

    def predict_pair(self, u_id, i_id, clip=False):#clip=True):
        """Возвращает прогноз рейтинга модели для данной пары пользователь - элемент.
        Args:
            u_id (int): идентификатор пользователя
            i_id (int): идентификатор элемента
            clip (boolean, default is `True`): стоит ли округлять прогноз или нет
        Returns:
            pred (float): рейтинг для данной пары пользователь - элемент
        """
        user_known, item_known = False, False
        pred = self.global_mean

        if u_id in self.user_dict:
            user_known = True
            u_ix = self.user_dict[u_id]
            pred += self.bu[u_ix]

        if i_id in self.item_dict:
            item_known = True
            i_ix = self.item_dict[i_id]
            pred += self.bi[i_ix]

        if  user_known and item_known:
            pred += np.dot(self.pu[u_ix], self.qi[i_ix])

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

    def predict(self, X):
        """Возвращает оценки нескольких заданных пар пользователь - элемент
        Args:
            X (pandas DataFrame): все пары пользователь - элемент, для которых мы хотим
                предсказывать рейтинги. Должен содержать столбцы `u_id` и `i_id`.
        Returns:
            predictions: список с прогнозами для пар пользователь - элемент
                pairs.
        """
        predictions = []

        for u_id, i_id in zip(X['u_id'], X['i_id']):
            predictions.append(self.predict_pair(u_id, i_id))

        return predictions

    def _early_stopping(self, list_val_rmse, min_delta=.001):
        """Возвращает True, если проверка rmse не улучшается
        Последнее значение rmse (плюс `min_delta`) сравнивается с предпоследним
        Agrs:
            list_val_rmse (list): список вычесленных ошибок RMSE
            min_delta (float, defaults to .001): малое значение дельты
        Returns:
            (boolean): стоит ли прекратить обучение
        """
        if list_val_rmse[-1] + min_delta > list_val_rmse[-2]:
            return True
        else:
            return False

    def _on_epoch_begin(self, epoch_ix):
        """Отображает журнал начала эпохи
        Args:
            epoch_ix (integer): индекс эпохи
        Returns:
            start (float): время начала текущей эпохи
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start

    def _on_epoch_end(self, start, val_loss=None, val_rmse=None, val_mae=None):
        """Отображение журнала окончания эпохи. Отображение метрик (val_loss/ val_rmse / val_mae).
        # Arguments
            start (float): время начала текущей эпохи
            val_loss (float): validation loss
            val_rmse (float): validation rmse
            val_mae (float): validation mae
        """
        end = time.time()

        print('val_loss: {:.3f}'.format(val_loss), end=' - ')
        print('val_rmse: {:.3f}'.format(val_rmse), end=' - ')
        print('val_mae: {:.3f}'.format(val_mae), end=' - ')

        print('took {:.1f} sec'.format(end - start))