import sys
=import os
import glob
import platform
import time
import datetime
from itertools import islice

class Timer(object):
  def __init__(self):
    self.t0 = time.time()
  
  def mark(self, msg=''):
    now = time.time()
    diff = now - self.t0
    self.t0 = now
    if len(msg):
      print('Elaspe: %f [%s] (%s)' % (diff, msg, datetime.datetime.now()))
    return diff

def datetime_to_timestr(dt):
  '''
  :param dt: datetime
  :return: '%Y/%m/%d %H:%M:%S'
  '''

  return dt.strftime('%Y/%m/%d %H:%M:%S')

#
def get_now_yyyymmdd():
  dt = datetime.datetime.now()
  return dt.strftime('%Y-%m-%d')

#
def datetimestr_to_epoch(datetimestr):
  dateval = datetime.datetime.strptime(datetimestr, "%Y/%m/%d %H:%M:%S")
  #print(datetimestr, dateval.timestamp())
  return dateval.timestamp()


def get_file_basename(fpath):
  fname = os.path.basename(fpath)
  try:
    basename = fname[:fname.index('.')]
  except:
    basename = fname
  
  if not platform.system() in ['Windows', 'Linux']: # Mac - 'Darwin'
    import unicodedata
    basename = unicodedata.normalize('NFC', basename)  # 맥에서 한글이름이 있는 파일명이 자소분리되는 현상을 해결하기 위한 것.
  return basename

  
##
class Tee(object):
  def __init__(self, name=None, mode='w'):
    if name is not None:
      self.file = open(name, mode)
    else:
      self.file = None
    self.stdout = sys.stdout
    # sys.stdout = self

  def __del__(self):
    sys.stdout = self.stdout
    self.file and self.file.close()
  
  def print(self, data):
    self.write(data)
  
  def write(self, data):
    if isinstance(data, bytes):
      data = data.decode('utf-8')

    self.file and self.file.write(data)
    self.stdout.write(data)
    self.flush()
  
  def flush(self):
    self.file and self.file.flush()
    self.stdout.flush()


##
class Log(object):
  def __init__(self, name, mode='w'):
    self.file = open(name, mode)

  def __del__(self):
    self.file.close()
    
  def write(self, data):
    self.file.write(data)
    self.file.flush()

    
##
def proc_one(fname, i):
  pass

##
def loop_in_folder(srcdir, func, limit=0, skip=0, logfile=None, sortBy='name'):
    
  if not platform.system() in ['Windows', 'Linux']: # Mac - 'Darwin'
    import unicodedata
    srcdir = unicodedata.normalize('NFD', srcdir) # denormalizing it since file names in Mac is denormalized
  print('SOURCE DIR', srcdir)
  files = glob.glob(srcdir)
  if sortBy == 'size':
    # 큰것부터
    files.sort(key=os.path.getsize, reverse=True)
  else:
    files.sort()
    
  loop_files(files, func, limit, skip, logfile)

##
def loop_files(files, func, limit=0, skip=0, logfile=None):
  print('SOURCE N=', len(files))
  if limit:
    print('!!! LIMIT - ', limit)
  if skip != 0:
    print('!!! SKIP - ', skip)

  
  log = Tee(logfile, mode='w')
    
  totalTimer = Timer()
  timer = Timer()
  for i, f in enumerate(files):
    if limit != 0 and i == limit:
      break
    if skip != 0 and i < skip:
      continue
      
    log and log.write('[%3d/%3d] %s ' %(i+1, len(files), f))
    res = func(f, i)
    if res:
      log and log.write('%s\t' % res)
    log and log.write(' %.3f sec\n' % timer.mark())
  
  # the end
  log and log.write('Total %.1f Seconds\n' % totalTimer.mark())

##


##
def iter_curr__next(iterable):
  '''
  :param iterable: ex) [0,1,2,3,4,5]
  :return: iterator on tuples of (current_element, next_element). ex) [(0,1), (1,2), (2,3), (3,4), (4,5)]
  '''
  return zip(islice(iterable, 0, len(iterable)-1), islice(iterable,1,len(iterable)))

  
if __name__=='__main__':
  iter = iter_curr__next(range(0,10,2))
  for i in range(10):
    print(next(iter))
  
