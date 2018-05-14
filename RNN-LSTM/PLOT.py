'''
animation gif 만들기

MAC OS 사전에 설치 필요한 사항


brew install yasm
brew install ffmpeg
brew install imagemagick
'''
import sys
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
import seaborn

##
def set_hangul():
  import matplotlib.font_manager as fm
  #[print(f.name, f.fname) for f in matplotlib.font_manager.fontManager.ttflist]
  hangulFonts = [(f.name, f.fname) for f in matplotlib.font_manager.fontManager.ttflist if 'Nanum' in f.name]
  #print(hangulFonts)
  if len(hangulFonts) == 0:
    fontprop = fm.FontProperties(fname="/usr/share/fonts/NanumGothic.ttf")
  else:
    fontprop = fm.FontProperties(fname=hangulFonts[0][1], size=9)
  return fontprop

FONT = set_hangul()


class LineChart(object):
  def __init__(self, xdata=None, miny=None, maxy=None):
    self.xdata = xdata
    
    self.miny = miny
    self.maxy = maxy
    self.fig = None
    self.pdf = None
    
  def _init_fig(self):
    if self.fig is None:
      self.fig = plt.figure()
      self.ax = plt.subplot()
      self.fig.set_tight_layout(True)
      
  def _close_fig(self):
    '''
    https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
    plt.cla() clears an axis,
    plt.clf() clears the entire current figure
    plt.close() closes a window
    '''
    #self.fig.clf()
    plt.close('all')
    self.fig = None
    
  def line(self, ydata, label=None):
    ''' Add one line to a page. A page can contain multiple lines '''
    self._init_fig()
    
    if self.miny is not None and self.maxy is not None:
      self.ax.set_ylim([self.miny, self.maxy])
    if self.xdata:
      self.ax.plot(self.xdata, ydata, label=label)
    else:
      self.ax.plot(ydata, label=label)
  
  def scatter(self, x, y, color, label=None):
    ''' Add one line to a page. A page can contain multiple lines '''
    self._init_fig()
    #self.ax.scatter(x, y, s=5, c=color, label=label)
    self.ax.scatter(x, y, c=color, label=label)
    
  def set_xticks(self, xvals, xnames):
    self.ax.xticks(xvals, xnames, rotation='horizontal', fontproperties=fontprop)
    #self.plt.xticks(xvals, xnames, rotation='horizontal', fontproperties=fontprop)
      
  def show(self, title=None):
    plt.legend(loc='upper left', prop=FONT)
    if title:
      plt.title(title, fontproperties=FONT)
    plt.show()
    self._close_fig()
    
  def save_png(self, fpath, title=None):
    ''' save a page into a PNG file'''
    plt.legend(loc='upper left', prop=FONT)
    if title:
      plt.title(title, fontproperties=FONT)
    self.fig.savefig(fpath)
    self._close_fig()
    print('Saved', fpath)

  def pdf_page(self, fpath, title=None):
    ''' insert a page into a pdf file. Yet, the pdf file is NOT complete.
        Call save_pdf()
    '''
    if self.pdf is None:
      self.pdf = PdfPages(fpath)
      self.pdffile = fpath
      
    plt.legend(loc='upper left', prop=FONT)
    if title:
      plt.title(title, fontproperties=FONT)
    self.pdf.savefig(self.fig)
    self._close_fig()
    
  def save_pdf(self, author=None):
    if self.pdf is None:
      return
    d = self.pdf.infodict()
    if author:
      d['Author'] = author
    d['CreationDate'] = datetime.datetime.now()
    self.pdf.close()
    print('Saved %s' % self.pdffile)
    
    
##
'''
pdf 파일로 생성
'''
class ToPdf(object):
  def __init__(self, fpath, title):
    self.fpath = fpath
    self.pdf = PdfPages(self.fpath)
    self.title = title
  
  def loop(self, data, labels, steps, miny, maxy, nframe=0):
    '''
    :param data: 출력할 데이터의 Y값 array의 목록. 즉 그래프가 복수개 출력 가능함. X값은 별도로 지정하지 않고 1,2,...,N (즉 인덱스값)
    :param labels: legend에 쓸 그래프 이름들의 목록
    :param steps: 한 화면에 출력할 데이터 개수
    :param miny: Y축 최소값
    :param maxy: Y축 최대값
    :param nframe: 최대 장수. 지정되지 않으면 data의 길이 / steps 에 의해 결정된다.
    :return:
    '''
    self.data = data
    self.steps = steps
    self.miny = miny
    self.maxy = maxy
  
    self.nframes = int(math.ceil(len(data[0]) / steps))
    if nframe > 0 and nframe < self.nframes:
      self.nframes = nframe
      
    for i in range(self.nframes):
      part = [ d[i * self.steps: (i + 1) * self.steps] for d in data]
      #print(i, len(part[0]))
      x = range(i * self.steps, (i + 1) * self.steps)
      self.add(x, part, labels, str(i))
    
  def add(self, x, ylist, labels, xlabel):
    '''
    한 페이지 출력
    :param x: X축값 목록.
    :param ylist:  Y축값
    :param labels: legend에 쓸 그래프 이름들의 목록
    :param xlabel: X축 아래에 쓸 텍스트
    :return:
    '''
  
    self.fig = plt.figure()
    self.ax = plt.subplot()
    self.fig.set_tight_layout(True)

    self.ax.set_ylim([self.miny, self.maxy])
    for i, y in enumerate(ylist):
      l = labels[i] if labels else None
      self.ax.set_xlabel(xlabel)
      self.ax.plot(x, y, label=l)
    plt.legend(loc='upper left', prop=FONT)
    plt.title(self.title, fontproperties=FONT)
    self.pdf.savefig(self.fig)
    plt.close()
    
  def save(self, author=None):
    d = self.pdf.infodict()
    if author:
      d['Author'] = author
    d['CreationDate'] = datetime.datetime.now()
    self.pdf.close()
    print('Saved %s' % self.fpath)


##
'''
Animation Gif 만들기
'''
class Anim(object):
  def __init__(self, data, steps, miny, maxy, delay=200, nframe=0):
    '''
    :param data: 출력할 데이터의 Y값 array의 목록. 즉 그래프가 복수개 출력 가능함. X값은 별도로 지정하지 않고 1,2,...,N (즉 인덱스값)
    :param steps: 한 화면에 출력할 데이터 개수
    :param miny: Y축 최소값
    :param maxy: Y축 최대값
    :param delay:
    :param nframe: 최대 장수. 지정되지 않으면 data의 길이 / steps 에 의해 결정된다.
    '''
    self.data = data
    self.steps = steps
    self.delay = delay
    self.miny = miny
    self.maxy = maxy
    
    self.fig = plt.figure()
    self.ax = plt.subplot()
    self.fig.set_tight_layout(True)
    self.nframes = int(math.ceil(len(data[0]) / steps))
    if nframe > 0 and nframe < self.nframes:
      self.nframes = nframe
    
    print('DATA', len(data), 'NFRAMES', self.nframes)
    self.anim = FuncAnimation(self.fig, self.update, frames=np.arange(self.nframes), interval=self.delay)
    
  def update(self, idx):
    #print('I', idx)
    label = 'timestep {0}'.format(idx)
    plt.cla()
    self.ax.set_ylim([self.miny, self.maxy])
    for d in self.data:
      line, = self.ax.plot(range(idx*self.steps, (idx + 1)*self.steps), d[idx*self.steps: (idx + 1)*self.steps], linewidth=2)
    # line.set_ydata()
    self.ax.set_xlabel(label)
  
  
  def save(self, fpath):
    self.anim.save(fpath, dpi=80, writer='imagemagick')
    print('Saved GIF', fpath)
    
  def show(self):
    plt.show()
    
if __name__ == '__main__':
  x = np.arange(0, 20, 0.01)
  r = [np.random.normal(0, 0.1, int(len(x)/20))]
  y = 3*np.sin(x) + np.hstack(r * 20)
  anim = Anim([y], 100, -10, 30, nframe=30, delay=200)
  #anim.show()
  anim.save('haha.gif')
  
  pdf = ToPdf('haha.pdf', '랜덤 Random data')
  pdf.loop([y], ['Y'], 100, -5, 10)
  pdf.save('Daehee')

  chart = LineChart(miny=-5, maxy=5)
  for i in range(0, len(y), 400):
    chart.line(y[i:i+400], 'Y')
    chart.pdf_page('test.pdf', title='%d' % i)
  chart.save_pdf('Daehee')
