"""
Links:
  [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
  [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import DATA

##
'''
predict_size 파라미터가 추가. 2017/10/31
'''
class LSTM:
    @staticmethod
    def RNN(x, weights, biases, windowsize, num_hidden):
      x = tf.unstack(x, windowsize, 1)
      lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden, forget_bias=1.0),
                                    rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)])
      outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
      
      return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
  
    def __init__(self, input_dim, window_size, num_hidden, output_dim,
                 predict_size=1,
                 name='lstm',
                 loss='square',
                 opt='grad'):
        '''
        LSTM모델의 입력차수, 윈도우 크기, hidden layer의 개수, 출력차수에 따라 모델구성 변수들을 생성하고,
        학습의 오차(loss) 계산식 정의, 최적화 함수 정의를 한다.

        :param input_dim:
        :param window_size:
        :param num_hidden:
        :param output_dim:
        :param predict_size:
        :param name:
        :param loss:
        :param opt:
        '''
        # Network Parameters
        self.input_dim = input_dim #
        self.window_size = window_size #
        self.num_hidden = num_hidden #
        self.output_dim = output_dim #
        self.predict_size = predict_size #

        self.valid_x = None
        self.valid_y = None
        self.valid_stop = 0 # 학습종료 판단하는 validation 기준값
        
        self.fig_num = 0 # for plotting

        # clear all things in tensorflow
        tf.reset_default_graph()
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.window_size, self.input_dim])
        self.Y = tf.placeholder("float", [None, self.predict_size * self.output_dim])
        
        # Define weights
        high = 4*np.sqrt(6.0/(self.num_hidden + self.output_dim))
        t = tf.Variable(tf.random_uniform([self.num_hidden, self.predict_size * self.output_dim], minval=-high, maxval=high, dtype=tf.float32))
        self.weights = {
            #'out': tf.Variable(tf.random_normal([self.num_hidden, self.output_dim]))
            # 참고 - https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-3-mnist_nn_xavier.py
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            # xavier initialization
            #'out': tf.get_variable("W", shape=[self.num_hidden, self.output_dim], initializer=tf.contrib.layers.xavier_initializer())
            'out': t
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.predict_size * self.output_dim]))
        }

        # self.prediction == logits
        self.prediction = LSTM.RNN(self.X, self.weights, self.biases, self.window_size, self.num_hidden)

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   200, 0.97, staircase=True)
        
        ## Define loss and optimizer
        # self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #  logits=self.logits, labels=self.Y))
        print('LOSS:', loss, ' OPT:', opt)
        if loss == 'abs':
          self.loss_op = tf.reduce_mean(tf.abs(self.prediction - self.Y))
        elif loss == 'softmax':
          self.loss_op = tf.reduce_mean(tf.nn.softmax(self.prediction))
        elif loss == 'softmax_entropy':
          self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              logits=self.prediction, labels=self.Y))
        else:
          self.loss_op = tf.reduce_mean(tf.square(self.prediction - self.Y))
        
        if opt == 'grad':
          self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif opt == 'adam':
          self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif opt == 'rms':
          self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
          assert(False)
          
        self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)
        
        # Evaluate model (with test logits, for dropout to be disabled)
        self.accuracy = self.loss_op
        self.name_network = self.build_model_name(name)
        self.training_stop = None
      
    def set_name(self, name):
        self.name_network = self.build_model_name(name)
  
    def build_model_name(self, name):
        return '%s-W%d-H%d-I%d-O%d' % \
               (name, self.window_size, self.num_hidden, self.input_dim, 
			self.output_dim * self.predict_size)
    
    def save(self):
        fname = '%s%s.ckpt' % (CFG.NNMODEL, self.name_network)
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, fname)
        print("Model saved in file: %s" % save_path)
    
    def load(self):
        fname = '%s%s.ckpt' % (CFG.NNMODEL, self.name_network)
        if not os.path.isfile(fname+'.index'):
          print('Model NOT found', fname)
          return False
        
        # Run the initializer
        self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, fname)
        print("Model restored from file: %s" % fname)
        return True
    
    def set_training_stop(self, training_stop):
      self.training_stop = training_stop
      
    def set_validation_data(self, valid_x, valid_y, valid_stop=0):
        if (valid_x is None) or (valid_x.shape[0] == 0):
          return
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.valid_stop = valid_stop # validation 결과값이 valid_stop 값 이하이면 학습 종료
      
    def do_validation(self):
        if self.valid_x is None or self.valid_x.shape[0] == 0:
          return 0
        valid_acc = self.do_test(self.valid_x, self.valid_y, 'Validation')
        return valid_acc
        
    
    def run(self, training_x, training_y, epochs=1000, batch_size=0, display_step=100):
        '''
        LSTM모델의 학습(training)을 수행하는 함수이다. training 데이터, validation 데이터를 별도로 지정할 수 있다.
        training 데이터 전체에서  batch_size만큼의 입력 및 출력 데이터를 1회의 batch training에 사용한다.
        batch를 일정 회수(display_step)만큼 수행한 후 validation 수치를 계산한다.
        학습 종료 조건으로 최대 epoch를 지정하거나,  validation목표 오차를 지정할 수 있다.

        :param training_x:
        :param training_y:
        :param epochs:
        :param batch_size:
        :param display_step:
        :return:
        '''
        self.max_epochs = epochs
        if batch_size == 0:
          batch_size = int(training_x.shape[0] * 0.05)
        self.display_step = display_step

        # Run the initializer
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        #training_y =  training_y.reshape((-1, self.predict_size * self.output_dim))
        training = DATA.BatchDataGen(training_x, training_y)
        
        for step in range(1, self.max_epochs+1):
            batch_x, batch_y = training.next_batch(batch_size)
            
            self.sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
            if step % self.display_step == 0 or step == 1:
                loss, acc = self.sess.run([self.loss_op, self.accuracy],
                                          feed_dict={self.X: batch_x, self.Y: batch_y})
                try:
                  curr_lr = self.sess.run(self.optimizer._learning_rate)
                except:
                  curr_lr = self.sess.run(self.optimizer._lr)
                
                print("Step " + str(step) + ": Acc= " + "{:.6f}".format(acc) + \
			                ", LR= " + "{:.6f}".format(curr_lr))

                if self.training_stop is not None and acc < self.training_stop:
                    print('STOP by training_stop')
                    break
                  
                valid_res = self.do_validation()
                if self.valid_stop != 0 and valid_res < self.valid_stop :
                    print('STOP by valid_stop')
                    break
        
        return acc, valid_res # training_error, validation_error
        
      
        
    def do_test(self, test_x, test_y, mesg='Test'):
        acc = self.sess.run(self.accuracy, feed_dict={self.X: test_x, self.Y: test_y})
        print("%s: %.6f" % (mesg, acc))
        return acc
    
    def do_compare(self, test_x, test_y):
        predict_y = self.sess.run(self.prediction, feed_dict={self.X: test_x, self.Y: test_y})
        diff = np.abs(predict_y - test_y) / test_y
        return diff

    def predict(self, test_x):
      predict_y = self.sess.run(self.prediction, feed_dict={self.X: test_x})
      #for a, p in zip(test_y[:20], predict_y[:20]):
      #  print('A, P', a, p)
      return predict_y
      
    def close(self):
      self.sess.close()
      



