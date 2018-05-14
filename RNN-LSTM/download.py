# data scrolling parts
import pandas as pd
import numpy as np
from pandas_datareader import data, wb
import datetime

JONGMOK = {

}

def save_one_from_yahoo(company, filename):
  start = datetime.datetime(2010, 1, 1)
  #end = datetime.datetime(2017, 12, 6) # 날짜 지정
  end = datetime.datetime.today() # 오늘 날짜
  
  df = data.DataReader(company, "yahoo", start, end)
  # df = data.get_data_yahoo(company) # DataReader() 기능과 유사.
  for col_name in df:
    print('COL:', col_name)
    df[col_name].fillna(value=0, inplace=True)
    #df[col_name] = df[col_name].astype(int)
    df[col_name] = df[col_name].map(lambda x: '%.2f' % x)
  
  df.to_csv(filename, header=True, index=True) # index가 날짜임
  print('Saved', filename)
  
if __name__ == '__main__':
  '''
  Yahoo Finance에서는
  KOSPI == '^KS11'
  Samsung Electronics Co., Ltd. (SSNLF)

  Google Finance에서는
  KOSPI = "KRX:KOSPI"
  삼성전자 == "KRX:005930"
  '''
  save_one_from_yahoo('^KS11', 'kospi.csv')
  save_one_from_yahoo('SSNLF', 'samsung.csv')
  save_one_from_yahoo('AMZN', 'amazon.csv')
  save_one_from_yahoo('AAPL', 'apple.csv')

