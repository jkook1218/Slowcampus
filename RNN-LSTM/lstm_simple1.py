import numpy as np

import DATA
import LSTM
import PLOT

'''
1차원 데이타의 시계열 LSTM 학습/테스트
'''

def lstm_with_sin():
  orgdata = np.linspace(0, 40, 1200, dtype=np.float32)
  sindata = np.sin(orgdata)
  
  train, validation, test = DATA.split_data(sindata)
  WINDOWSIZE = 80
  train_x, train_y = DATA.getSeriesData(train, WINDOWSIZE)
  valid_x, valid_y = DATA.getSeriesData(validation, WINDOWSIZE)
  test_x, test_y = DATA.getSeriesData(test, WINDOWSIZE)
  
  print('TRAIN', train.shape)
  print('TEST', test.shape)
  print('TRAIN X', train_x.shape)
  print('TRAIN Y', train_y.shape)
  
  lstm = LSTM.LSTM(1, WINDOWSIZE, 4, 1, loss='square', opt='rms')
  lstm.set_validation_data(valid_x, valid_y, valid_stop=0.0001)
  lstm.run(train_x, train_y, batch_size=100, epochs=1000)
  lstm.do_test(test_x, test_y)
  predict_y = lstm.predict(test_x)
  chart = PLOT.LineChart()
  chart.line(test_y, 'Actual')
  chart.line(predict_y, 'ByNN')
  chart.show()


if __name__ == '__main__':
  lstm_with_sin()


