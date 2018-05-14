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
import LSTMAE
import PLOT

''' Autoencoder  방식의 학습 '''
#
def normalize_DTG(df):
  df.SP = df.SP / 100.0
  df.RPM = df.RPM / 1000.0
  #df.X = df.X - 127.01
  df.Y = (df.Y - 37.13) / 0.4 # 오산~한남 위도차 == 0.4
  #df.ROAD_EV = df.ROAD_EV - df.ROAD_EV.shift(-3)
  #print('ROAD EV DIFF MAX', df.ROAD_EV.max())
  #df.ROAD_EV = df.ROAD_EV /  df.ROAD_EV.max()
  #df.BRAKE = df.BRAKE.map(lambda x: 1 if x == 1 else -1)
  #df.DIR = df.DIR.map(lambda x: 1 if x=='E' else -1)
  #df.BUS_SPEED = df.BUS_SPEED.where(df.BUS_SPEED > 0, df.GEN_SPEED)
  #df.BUS_SPEED = df.BUS_SPEED / 100.0
  df.fillna(0, inplace=True)
  
def denormalize_DTG_DF(df):
  df.SP = df.SP * 100.0
  df.RPM = df.RPM * 1000.0
  #df.X = df.X + 127.01
  df.Y = df.Y * 0.4 + 37.13
  #df.ROAD_EV = df.ROAD_EV * 100.0
  #if 'PR_SP' in df.columns:
  #  df.PR_SP = df.PR_SP * 100.0
  #if 'PR_RPM' in df.columns:
  #  df.PR_RPM = df.PR_RPM * 1000.0
  #df.BUS_SPEED = df.BUS_SPEED * 100.0
  
def denormalize_DTG_elem(elem):
  for i in range(int(len(elem) / 4)):
    elem[i*4 + 0] = elem[i*4 + 0] * 100.0
    elem[i*4 + 1] = elem[i*4 + 1] * 1000.0
    elem[i*4 + 3] = elem[i*4 + 3] * 0.4 + 37.13
  return elem

def denormalize_DTG_array(darr):
  #np.apply_along_axis(denormalize_DTG_elem, 0, darr)
  for e in darr:
    denormalize_DTG_elem(e)

##
def onehot(dim, vals):
  hot = np.eye(2)[np.array(vals.reshape((-1,1)), dtype=int)]
  return hot.reshape((-1,2))

#
def timeseries_for_onetrip(df, s1, s2, window_size):
  onetrip = df[s1:s2][['SP', 'RPM', 'BRAKE', 'X', 'Y', 'ROAD_EV']].as_matrix()
  # print(onetrip[0], onetrip[-1])
  #print(onetrip.shape, s2 - s1)
  x, y = LSTMAE.getSeriesData(onetrip, window_size, elementdim=onetrip.shape[1])
  #x, y = LSTMAE.getInBetweenData(onetrip, window_size, elementdim=onetrip.shape[1])
  return x, y

#
def conv_y_data(orgy, colname):
  idx = FACTORS.index(colname)
  #multiidx = [i * INPUT_DIM + idx for i in range(PREDICTSIZE)]
  return orgy[:, :, idx].reshape((-1, PREDICTSIZE))
  
##
def do_prediction_compare(lstm, test_x, test_y, name):
  '''
  output Y의 Actual값과 예측값 비교
  '''
  actual = test_y
  predicted = lstm.predict(test_x)
  # convert to 1d data
  print('SHAPE ACTUAL', actual.shape)
  print('SHAPE PREDICTED', predicted.shape)
  # actual = np.array([a.mean() for a in actual])
  # predicted = np.array([a.mean() for a in predicted])
  # print('SHAPE ACTUAL', actual.shape)
  # print('SHAPE PREDICTED', predicted.shape)
  #actual = actual.reshape((-1))
  #predicted = predicted.reshape((-1))
  
  return actual, predicted

def do_save_vals_of_prediction(actual, predicted, name, fname):
  #merge = np.stack((actual, predicted), axis=1)
  #df = pd.DataFrame(merge)
  fpath = CFG.NNPREDICT+'predict_%s_%s.csv' % (fname, name)
  #df.to_csv(fpath, sep=',', header=False, index=False)
  with open(fpath, 'wt') as out:
    i = 0
    for a,p in zip(actual, predicted):
      out.write(','.join(['%.1f' % v for v in a]))
      out.write('\n')
      out.write(','.join(['%.1f' % v for v in p]))
      out.write('\n')
      i += 1
      if i >= 1000:
        break
    

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

def do_save_big_diff_df(df, fname):
  #print('NULL NEW DF', df.isnull().sum().sum())
  denormalize_DTG_DF(df)
  df.X = df.X.map(lambda x: '%.6f' % x)
  df.Y = df.Y.map(lambda x: '%.6f' % x)
  df.SP = df.SP.map(lambda x: '%.1f' % x)
  df.RPM = df.RPM.map(lambda x: '%.1f' % x)
  fpath = CFG.NNPREDICT + 'big_diff_df_%s.csv' % (fname)
  df.to_csv(fpath, sep=',', index=False, header=True)
  print('Saved %s' % fpath)

##
def calc_diff_per_elem(actual, predict, elem_dim):
  
  a = actual.reshape((actual.shape[0], -1, elem_dim))
  p = predict.reshape((predict.shape[0], -1, elem_dim))
  d = p - a
  
  # print(actual[0])
  # print(a[0])
  # print(predict[0])
  # print(p[0])
  #print(d[0])
  print('DIFF2', np.average(np.absolute(d[0]), axis=0))
  
  # np.sum(np.square(x), axis=0) --> SP, RPM, BRAKE, Y 에 대한 각각별 오차
  # 각 window별 SP, RPM, BRAKE, Y 에 대한 각각별 오차
  avg_of_abs_per_elem = np.apply_along_axis(lambda x: np.average(np.absolute(x), axis=0), axis=1, arr=d)
  avg_per_elem = np.apply_along_axis(lambda x: np.average(x, axis=0), axis=1, arr=d)
  #print('DIFF', avg_of_abs_per_elem[:10])
  return avg_of_abs_per_elem, avg_per_elem
  
  
#### LSTM 모델 및 input 변수
WINDOWSIZE = 20
FACTORS = ['SP', 'RPM', 'BRAKE', 'Y']
INPUT_DIM = len(FACTORS)
OUTPUT_DIM = INPUT_DIM
HIDDEN_NODES = 20
PREDICTSIZE = WINDOWSIZE  #

##
def train_autoencoder_for_dtg(file, i):
  timer = DTG.Timer()
  fname = DTG.get_file_basename(file)
  df = pd.read_csv(file, names=DTG.BUS_DTG_FLDNAMES)
  timer.mark('Loading')
  print('ALL', df.TIME.count())
  #df.TIME = df.TIME.map(DTG.datetimestr_to_epoch)
  df.EPOCH = df.TIME.map(lambda t: DTG.dtgtimestr_to_epoch('%010d' % t))
  df.TMDIFF = df.EPOCH.shift(-WINDOWSIZE) - df.EPOCH
  df = df[df.TMDIFF == WINDOWSIZE]
  ## reset_index() 아주 중요. 해결하는데 4시간 소요. index를 재정렬해야 새 컬럼 추가시에 연결이 잘된다
  df.reset_index(inplace=True, drop=True)

  print('VALID', df.TIME.count())
  print('NULL', df.isnull().sum().sum())
  #timer.mark('Valid trip')

  ##
  normalize_DTG(df)
  data = df[FACTORS].values
  #print(data[:10])
  num = len(data) - WINDOWSIZE + 1
  training_x = np.array([data[i:i + WINDOWSIZE] for i in range(num)], dtype=np.float32)
  training_y = training_x.reshape((-1, PREDICTSIZE * OUTPUT_DIM))
  print('SHAPE X', training_x.shape)
  timer.mark('Ready for NN')
  ##
  name = 'AE'
  lstm = LSTMAE.LSTM(INPUT_DIM, WINDOWSIZE, HIDDEN_NODES, OUTPUT_DIM, PREDICTSIZE,
                     name='AE', opt='adam', loss='square')
  lstm.set_name('lstm_%s_%s.net' % (fname, name))
  nn_loaded = lstm.load()
  if not nn_loaded:
    lstm.set_training_stop(0.0005)
    training_acc, valid_acc = lstm.run(training_x, training_y, epochs=5000, batch_size=500)
    lstm.save()
    
  timer.mark('After NN')
  
  actual, predicted = do_prediction_compare(lstm, training_x, training_y, name)
  avg_of_abs_per_elem, avg_per_elem = calc_diff_per_elem(actual, predicted, INPUT_DIM)
  print('SHAPE DIFF', avg_of_abs_per_elem.shape)
  timer.mark('Calc Diff')

  #save_diff_1(df, actual, predicted, fname, name)
  # 예측결과값을 원본 DataFrame의 칼럼으로 추가하기 위한 것.
  # colname = 'PR_%s' % (name)
  # filling = np.full((WINDOWSIZE), np.NaN) # fill with NaN
  # df[colname] = pd.Series(np.hstack([filling, predicted]))
  # do_save_org_df(df, fname)
  df['SP_ERR'] = pd.Series(avg_per_elem[:,0])
  df['RPM_ERR'] = pd.Series(avg_per_elem[:, 1])
  df['BR_ERR'] = pd.Series(avg_per_elem[:, 2])
  df['SP_ABS_ERR'] = pd.Series(avg_of_abs_per_elem[:, 0])
  df['RPM_ABS_ERR'] = pd.Series(avg_of_abs_per_elem[:, 1])
  df['BR_ABS_ERR'] = pd.Series(avg_of_abs_per_elem[:, 2])
  #print(df[:10])

  denormalize_DTG_DF(df)
  df.X = df.X.map(lambda x: '%.6f' % x)
  df.Y = df.Y.map(lambda x: '%.6f' % x)
  df.SP = df.SP.map(lambda x: '%.1f' % x)
  df.RPM = df.RPM.map(lambda x: '%.1f' % x)
  df.SP_ERR = df.SP_ERR.map(lambda x: '%.6f' % x)
  df.RPM_ERR = df.RPM_ERR.map(lambda x: '%.6f' % x)
  df.BR_ERR = df.BR_ERR.map(lambda x: '%.6f' % x)
  
  df.SP_ABS_ERR = df.SP_ABS_ERR.map(lambda x: '%.6f' % x)
  df.RPM_ABS_ERR = df.RPM_ABS_ERR.map(lambda x: '%.6f' % x)
  df.BR_ABS_ERR = df.BR_ABS_ERR.map(lambda x: '%.6f' % x)
  fpath = CFG.NNPREDICT + 'predict_diff_%s_%s.csv' % (fname, name)
  print('Saving', fpath)
  df.to_csv(fpath, sep=',', header=True, index=False,
            columns=['SEQ', 'TIME', 'X', 'Y', 'SP','RPM', 'BRAKE' \
                     ,'SP_ERR', 'RPM_ERR', 'BR_ERR' \
                     ,'SP_ABS_ERR', 'RPM_ABS_ERR', 'BR_ABS_ERR'
                     ])


  

#
def save_diff_1(df, actual, predicted, fname, name):
  # see = [(i, np.linalg.norm(d)) for i,d in enumerate(diff)]
  # see.sort(key=lambda x: x[1], reverse=True)
  fpath = CFG.NNPREDICT + 'predict_diff_%s_%s.csv' % (fname, name)
  print('Save', fpath)
  
  diff = actual - predicted
  see = [(i, np.linalg.norm(d)) for i,d in enumerate(diff)]
  see.sort(key=lambda x: x[1], reverse=True)

  # print(see[:10])
  # denormalize_DTG_array(actual)
  # denormalize_DTG_array(predicted)
  
 
  with open(fpath, 'wt') as out:
    for i in range(100):
      pos = see[i][0]
      dd = ['%.2f' % v for v in diff[pos]]
      aa = ['%.1f' % v for v in actual[pos]]
      pp = ['%.1f' % v for v in predicted[pos]]
      out.write('[%d] (%d) %.2f\n' % (i, pos, see[i][1]))
      out.write('D %s\n' % ' '.join(dd))
      out.write('A %s\n' % ' '.join(aa))
      out.write('P %s\n\n' % ' '.join(pp))
      
  index_list = [see[i][0] for i in range(200)]
  seedf = df.iloc[index_list].copy()
  do_save_big_diff_df(seedf, fname)
  #print(seedf)
  #do_save_vals_of_prediction(actual, predicted, name, fname)
  #do_save_img_of_prediction(actual, predicted, name, fname)
  
  # 예측결과값을 원본 DataFrame의 칼럼으로 추가하기 위한 것.
  # colname = 'PR_%s' % (name)
  # filling = np.full((WINDOWSIZE), np.NaN) # fill with NaN
  # df[colname] = pd.Series(np.hstack([filling, predicted]))
  # do_save_org_df(df, fname)
  
 

###
if __name__ == '__main__':
  DTG.loop_in_folder(CFG.DTG_JOINED+'*.csv', train_autoencoder_for_dtg,
                     sortBy='size', limit=1, skip=0, logfile='log.train_nn.3sec')


  