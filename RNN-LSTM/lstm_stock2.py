import numpy as np
import pandas as pd
import DATA
import LSTM
import PLOT

'''
2차원 데이타의 시계열 LSTM 학습/테스트
'''


def MinMaxScaler(data):
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  # noise term prevents the zero division
  return numerator / (denominator + 1e-7)


def lstm_stock(fname):
  df = pd.read_csv(fname, header=0)
  print(df.columns)
  MAX_PRICE = df.Open.max()
  print('MAX_PRICE', MAX_PRICE)
  df = df[df.Open != 0][['Open', 'Close']]
  df.Open = df.Open / MAX_PRICE
  df.Close = df.Close / MAX_PRICE
  xy = df.as_matrix()
  #xy = MinMaxScaler(xy)
  
  train, validation, test = DATA.split_data(xy)
  WINDOWSIZE = 60
  train_x, train_y = DATA.getSeriesData(train, WINDOWSIZE, elementdim=2)
  valid_x, valid_y = DATA.getSeriesData(validation, WINDOWSIZE, elementdim=2)
  test_x, test_y = DATA.getSeriesData(test, WINDOWSIZE, elementdim=2)
  
  print('TRAIN', train.shape)
  print('TEST', test.shape)
  print('TRAIN X', train_x.shape)
  print('TRAIN Y', train_y.shape)
  
  
  lstm = LSTM.LSTM(2, WINDOWSIZE, 2, 2, loss='square', opt='adam')
  lstm.set_validation_data(valid_x, valid_y, valid_stop=0.0001)
  lstm.run(train_x, train_y, batch_size=int(train_x.shape[0] / 20), epochs=1000)
  lstm.do_test(test_x, test_y)
  predict_y = lstm.predict(test_x)
  chart = PLOT.LineChart()
  chart.line(test_y[:, 0], 'Actual')
  chart.line(predict_y[:, 0], 'ByNN')
  chart.show()
  
  chart = PLOT.LineChart()
  chart.line(test_y[:, 1] * MAX_PRICE, 'Actual')
  chart.line(predict_y[:, 1] * MAX_PRICE, 'ByNN')
  chart.show()
  


if __name__ == '__main__':
  lstm_stock('samsung.csv')

