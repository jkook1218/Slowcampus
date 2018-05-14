import numpy as np
'''
1) NN training을 위한 batch데이터 생성 - BatchDataGen()
2) LSTM training을 위한 연속데이터 생성 - getSeriesData()
'''

###
class BatchDataGen(object):
  '''
  this is a simplifed version of 'DataSet' class from a tensorflow's source file,
  '/tensorflow/contrib/learn/python/learn/datasets/mnist.py'
  데이터 목록에서 요청된 batch size만큼 데이터를 묶어서 제공한다.
  circular하게 처리하여 데이터 제공이 무한 루프하도록 한다.

  '''
  
  def __init__(self, x, y):
    self.x = x
    self.y = y
    assert x.shape[0] == y.shape[0], ('x.shape: %s y.shape: %s' % (x.shape, y.shape))
    self._num_examples = x.shape[0]
    self._epochs_completed = 0
    self._index_in_epoch = 0
    print('DataSet X', x.shape)
    print('DataSet Y', y.shape)
    print('DataSet num_examples', self._num_examples)
  
  def next_batch(self, batch_size, shuffle=True):
    assert batch_size <= self._num_examples, ('batch_size %d <= num_examples %d' % (batch_size, self._num_examples))
    # print('BATCH %d, Start %d' % (batch_size, self._index_in_epoch))
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self.x = self.x[perm0]
      self.y = self.y[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      x_rest_part = self.x[start:self._num_examples]
      y_rest_part = self.y[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.y = self.y[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      x_new_part = self.x[start:end]
      y_new_part = self.y[start:end]
      return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self.x[start:end], self.y[start:end]


## Generator of Data for RNN. Time series data
class SeriesDataGen(object):
  def __init__(self, _data, _window_size):
    self.data = _data
    self.idx = 0
    self.window_size = _window_size
    assert self.window_size < len(self.data)
  
  def __next__(self):
    end = self.idx + self.window_size
    if end >= len(self.data):
      self.idx = 0
      end = self.window_size
    
    next_x = self.data[self.idx:end]
    next_y = self.data[end]
    self.idx += 1
    return next_x, next_y


## LSTM 학습용을 위한 데이터 생성. window개념 포함.
def getSeriesData(data, window_size, elementdim=1, predict_size=1):
  '''
  window_size 만큼의 데이터로 다음 데이터를 예측하기 위한 학습데이터 생성
  :param data: [1,2,3,4,5]
  :param window_size:
  :param elementdim:
  :return: x=[1,2,3], y=[4], x=[2,3,4], y=[5]
  '''
  assert window_size < len(data)
  num = len(data) - window_size - predict_size + 1
  xdata = [data[i:i + window_size] for i in range(num)]
  ydata = [data[i + window_size:i + window_size + predict_size] for i in range(num)]
  
  x = np.array(xdata, dtype=np.float32).reshape((-1, window_size, elementdim))
  y = np.array(ydata, dtype=np.float32).reshape((-1, predict_size * elementdim))
  return x, y


## 주소 - https://github.com/akash13singh/LSTM_TimeSeries
def split_data(data, val_size=0.15, test_size=0.15):
  """
  splits data to training, validation and testing parts
  """
  ntest = int(round(len(data) * (1 - test_size)))
  nval = int(round(ntest * (1 - val_size)))
  
  train, validation, test = np.split(data, [nval, ntest])
  
  return train, validation, test


##
def split_list(lst, val_size=0.15, test_size=0.15):
  """
  splits data to training, validation and testing parts
  """
  # total = df.shape[0]
  total = len(lst)
  ntest = int(round(total * (1 - test_size)))
  nval = int(round(ntest * (1 - val_size)))
  
  train = lst[:nval]
  validation = lst[nval:ntest]
  test = lst[ntest:]
  return train, validation, test


##
def test_generator():
  d = np.random.rand(10, 1)
  d = np.arange(1,2,0.1)
  print(d)
  x, y = getSeriesData(d, 5)
  print(x)
  print(y)
  print('-----')
  d = SeriesDataGen(d, 5)
  for i in range(10):
    print(next(d))
    
if __name__ == '__main__':
  test_generator()