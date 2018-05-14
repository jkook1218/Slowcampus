import numpy as np

import DATA
import LSTM
import PLOT

'''
2차원 데이타의 시계열 LSTM 학습/테스트
'''

def lstm_with_sin_cos():
  NUM_DATA = 1200
  orgdata = np.linspace(0, 40, NUM_DATA, dtype=np.float32)
  sindata = np.sin(orgdata) + 0.1
  cosdata = np.cos(orgdata) * 2
  merge = np.stack((sindata, cosdata), axis=-1)
  print(merge[:3])
  
  train, validation, test = DATA.split_data(merge)
  WINDOWSIZE = 40
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
  chart.scatter(test_y[:, 0], test_y[:, 1], 'g', 'Actual')
  chart.scatter(predict_y[:, 0], predict_y[:, 1], 'r', 'ByNN')
  chart.show()
  


if __name__ == '__main__':
  lstm_with_sin_cos()

