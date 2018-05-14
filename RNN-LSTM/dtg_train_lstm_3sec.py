import sys
import collections
import operator
import itertools
import pandas as pd
import numpy as np
try:
	from matplotlib import pyplot as plt
except:
	pass
import os
import DTG, CFG
import LSTM2 as LSTM

#
def normalize_DTG(df):
  df.SP = df.SP / 100.0
  df.RPM = df.RPM / 1000.0
  #df.X = df.X - 127.01
  #df.Y = (df.Y - 37.13) / 0.4 # 오산~한남 위도차 == 0.4
  df.ROAD_EV = df.ROAD_EV - df.ROAD_EV.shift(-3)
  print('ROAD EV DIFF MAX', df.ROAD_EV.max())
  df.ROAD_EV = df.ROAD_EV /  df.ROAD_EV.max()
  #df.BRAKE = df.BRAKE.map(lambda x: 1 if x == 1 else -1)
  #df.DIR = df.DIR.map(lambda x: 1 if x=='E' else -1)
  df.BUS_SPEED = df.BUS_SPEED.where(df.BUS_SPEED > 0, df.GEN_SPEED)
  df.BUS_SPEED = df.BUS_SPEED / 100.0

  df.ROAD_EV.fillna(0, inplace=True)
  
def denormalize_DTG_DF(df):
  df.SP = df.SP * 100.0
  df.RPM = df.RPM * 1000.0
  #df.X = df.X + 127.01
  #df.Y = df.Y + 37.13
  #df.ROAD_EV = df.ROAD_EV * 100.0
  
  #if 'PR_SP' in df.columns:
  #  df.PR_SP = df.PR_SP * 100.0
  #if 'PR_RPM' in df.columns:
  #  df.PR_RPM = df.PR_RPM * 1000.0
  
  df.BUS_SPEED = df.BUS_SPEED * 100.0
  
def denormalize_DTG_np(arr):
  arr[0] = arr[0] * 100.0
  arr[1] = arr[1] * 1000.0
  if len(arr) >= 4:
    arr[3] = arr[3] + 127.01
    arr[4] = arr[4] + 37.13
    arr[5] = arr[5] * 100.0
  return arr

def denormalize_DTG_np_nd(darr):
  np.apply_along_axis(denormalize_DTG_np, 0, darr)

def denormalize_DTG_np_1d(one, name):
  if name == 'SP':
    return one * 100
  elif name == 'RPM':
    return one * 1000

##
def onehot(dim, vals):
  hot = np.eye(2)[np.array(vals.reshape((-1,1)), dtype=int)]
  return hot.reshape((-1,2))

#
def timeseries_for_onetrip(df, s1, s2, window_size):
  onetrip = df[s1:s2][['SP', 'RPM', 'BRAKE', 'X', 'Y', 'ROAD_EV']].as_matrix()
  # print(onetrip[0], onetrip[-1])
  #print(onetrip.shape, s2 - s1)
  x, y = LSTM.getSeriesData(onetrip, window_size, elementdim=onetrip.shape[1])
  #x, y = LSTM.getInBetweenData(onetrip, window_size, elementdim=onetrip.shape[1])
  return x, y

#
def conv_y_data(orgy, colname):
  idx = FACTORS.index(colname)
  #multiidx = [i * INPUT_DIM + idx for i in range(PREDICTSIZE)]
  return orgy[:, :, idx].reshape((-1, PREDICTSIZE))
  
##
## LSTM 모델 및 input 변수
WINDOWSIZE = 30
FACTORS = ['SP', 'RPM', 'BRAKE', 'BUS_SPEED', 'ROAD_EV']
INPUT_DIM = len(FACTORS)
HIDDEN_NODES = 10
PREDICTSIZE = 3  # 3초간의 속도, RPM을 예측.

##
def prepare_data_simple(df):
  normalize_DTG(df)
  data = df[FACTORS].values
  
  x, y = LSTM.getSeriesData(data, WINDOWSIZE, elementdim=data.shape[1], predict_size=PREDICTSIZE)
  training_x, validation_x, test_x = LSTM.split_data(x, val_size=0.0, test_size=0.0)
  training_y, validation_y, test_y = LSTM.split_data(y, val_size=0.0, test_size=0.0)
  return training_x, training_y, validation_x, validation_y, test_x, test_y 
  
##
def do_train_rnn(training_x, training_y, validation_x, validation_y, name, plot=False, fname=None):
  print('%s -- TRAINING: %d, VALIDATION: %d' % (name, training_x.shape[0],  validation_x.shape[0]))
  
  training_y = conv_y_data(training_y, name)
  output_dim = 1
    
  lstm = LSTM.LSTM(INPUT_DIM, WINDOWSIZE, HIDDEN_NODES, output_dim, PREDICTSIZE, name='NEXT3', opt='adam')
  lstm.set_name('lstm_%s_%s.net' % (fname, name))
  #lstm.set_validation_data(validation_x, validation_y, valid_stop=0.0002 if name=='Brake' else 0.0005)
  
  nn_loaded = lstm.load()
  if not nn_loaded:
    lstm.set_training_stop(0.0001)
    training_acc, valid_acc = lstm.run(training_x, training_y, epochs=2000, batch_size=400)
    lstm.save()
  return lstm


##
def do_prediction_compare(lstm, test_x, test_y, name):
  '''
  output Y의 Actual값과 예측값 비교
  '''
  actual = conv_y_data(test_y, name)
  predicted = lstm.predict(test_x)
  # convert to 1d data
  print('SHAPE ACTUAL', actual.shape)
  print('SHAPE PREDICTED', predicted.shape)
  actual = np.array([a.mean() for a in actual])
  predicted = np.array([a.mean() for a in predicted])
  print('SHAPE ACTUAL', actual.shape)
  print('SHAPE PREDICTED', predicted.shape)
  #actual = actual.reshape((-1))
  #predicted = predicted.reshape((-1))
  a = denormalize_DTG_np_1d(actual, name)
  p = denormalize_DTG_np_1d(predicted, name)
  return a, p

def do_save_vals_of_prediction(actual, predicted, name, fname):
  #merge = np.stack((actual, predicted), axis=1)
  #df = pd.DataFrame(merge)
  fpath = CFG.NNPREDICT+'predict_%s_%s.csv' % (fname, name)
  #df.to_csv(fpath, sep=',', header=False, index=False)
  with open(fpath, 'wt') as out:
    for a,p in zip(actual, predicted):
      out.write('%.1f,%.1f\n' % (a, p))

  print('Saved %s' % fpath)

def do_save_img_of_prediction(actual, predicted, name, fname):
  try:
    fig = plt.figure()
  except:
    return
  ax = plt.subplot(111)
  
  ax.plot(actual[5000:5500], label='Actual')
  ax.plot(predicted[5000:5500], label='ByNN')
  ax.legend()
  title = '%s - %s' % (fname, name)
  plt.title(title)
  fpath = CFG.IMG+'%s-%s.png' % (fname, name)
  fig.savefig(fpath)
  plt.close(fig)
  print('Saved %s' % fpath)

def do_save_org_df(df, fname):
  #print('NULL NEW DF', df.isnull().sum().sum())
  denormalize_DTG_DF(df)
  df.X = df.X.map(lambda x: '%.6f' % x)
  df.Y = df.Y.map(lambda x: '%.6f' % x)
  df['SP_ERR'] = df.SP - df.PR_SP
  df['RPM_ERR'] = df.RPM - df.PR_RPM
  df.ROAD_EV = df.ROAD_EV.map(lambda x: '%.1f' % x)
  df.BUS_SPEED = df.BUS_SPEED.map(lambda x: '%.1f' % x)
  df.SP = df.SP.map(lambda x: '%.1f' % x)
  df.RPM = df.RPM.map(lambda x: '%.1f' % x)
  df.PR_SP = df.PR_SP.map(lambda x: '%.1f' % x)
  df.PR_RPM = df.PR_RPM.map(lambda x: '%.1f' % x)
  df.SP_ERR = df.SP_ERR.map(lambda x: '%.1f' % x)
  df.RPM_ERR = df.RPM_ERR.map(lambda x: '%.1f' % x)
  
  fpath = CFG.NNPREDICT + 'lstm_df_%s.csv' % (fname)
  df.to_csv(fpath,
            columns=['SEQ', 'TIME', 'X', 'Y', 'ROAD_EV', 'DIR', 'BRAKE', 'BUS_SPEED', 'SP', 'PR_SP', 'SP_ERR', 'RPM',
                     'PR_RPM', 'RPM_ERR'],
            sep=',', index=False)
  print('Saved %s' % fpath)
  
##
def timeseries_from_dtg(file, i):
  fname = DTG.get_file_basename(file)
  df = pd.read_csv(file, names=DTG.BUS_DTG_FLDNAMES)
  print('ALL', df.TIME.count())
  
  timer = DTG.Timer()
  #df.TIME = df.TIME.map(DTG.datetimestr_to_epoch)
  df.TIME = df.TIME.map(lambda t: DTG.dtgtimestr_to_epoch('%010d' % t))
  df.TMDIFF = df.TIME.shift(-WINDOWSIZE) - df.TIME
  
  df = df[df.TMDIFF == WINDOWSIZE]
  ## reset_index() 아주 중요. 해결하는데 4시간 소요. index를 재정렬해야 새 컬럼 추가시에 연결이 잘된다
  df.reset_index(inplace=True, drop=True)

  print('VALID DF', df.TIME.count())
  print('NULL', df.isnull().sum().sum())
  #timer.mark('TIME conv')
  data_for_nn = prepare_data_simple(df)
  #timer.mark('DATA Prepared')
  
  if not data_for_nn:
    return 

  training_x, training_y, validation_x, validation_y, test_x, test_y = data_for_nn
  print('SHAPE X', training_x.shape)
  print('SHAPE Y', training_y.shape)
  #print('%d, %d, %d' % (training_x.shape[0], validation_x.shape[0], test_x.shape[0]))
  names = ['SP', 'RPM']
  for name in names:
    lstm = do_train_rnn(training_x, training_y,
                        validation_x, validation_y,
                        name,
                        fname=fname)
    
    actual, predicted = do_prediction_compare(lstm, training_x, training_y, name)
    #do_save_vals_of_prediction(actual, predicted, name, fname)
    #do_save_img_of_prediction(actual, predicted, name, fname)
    
    # 예측결과값을 원본 DataFrame의 칼럼으로 추가하기 위한 것.
    colname = 'PR_%s' % (name)
    filling = np.full((WINDOWSIZE), np.NaN) # fill with NaN
    # WINDOWSIZE 만큼 shift시키는 것임.
    new_col_np = np.hstack([filling, predicted])
    #print('NEW_COL_NP', new_col_np.shape[0], new_col_np[-5:])
    new_col_sr = pd.Series(new_col_np)
    #print('NEW_COL_SR', new_col_sr.shape[0], new_col_sr[-5:])
    df[colname] = new_col_sr
    #print('NEW_COL_SR', df[colname].shape[0], df[colname][-5:])
  do_save_org_df(df, fname)
  
 

###
if __name__ == '__main__':
  DTG.loop_in_folder(CFG.DTG_JOINED+'*.csv', timeseries_from_dtg,
                     sortBy='size', limit=0, skip=0, logfile='log.train_nn.3sec')
