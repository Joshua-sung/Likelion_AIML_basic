#예제문제
data=[[0.3,12.27],[-0.78,14.44],[1.26,11.87],[0.03,18.75],[1.11,17.52],[0.24,16.37],[-0.24,19.78],[-0.47,19.51],
      [-0.77,12.65],[-0.37,14.74],[-0.85,10.72],[-0.41,21.94],[-0.27,12.83],[0.02,15.51],[-0.76,17.14],[2.66,14.42]]

inc=[i[0] for i in data]
old=[i[1] for i in data]
 
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np

a=tf.Variable(tf.random_uniform([1],0,10, dtype=tf.float64,seed=0))
b=tf.Variable(tf.random_uniform([1],0,100, dtype=tf.float64,seed=0))
y=a*inc+b 
rmse=tf.sqrt(tf.reduce_mean(tf.square(y-old)))
learning_rate=0.1
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(2001):
    sess.run(gradient_decent)
    if step %100==0:
      print('Epoch: %.f,RMSE=%.04f, 기울기 a=%.4f, y절편 b=%.4f'%(step,sess.run(rmse),sess.run(a),sess.run(b)))
  data_a=sess.run(a)
  data_b=sess.run(b)
line_x=np.arange(min(inc),max(inc),0.01)
line_y=data_a*line_x+data_b
plt.plot(line_x,line_y, c='r', lw=3, ls='-',marker='o',ms=5,mfc='b')
plt.plot(inc,old,'bo')
plt.xlabel('Population Growth Rate(%)')
plt.ylabel('Elderly Growth Rate(%)')
plt.show()

#다중 선형 회귀
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
data=[[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
x1=[x_row1[0] for x_row1 in data]
x2=[x_row2[1] for x_row2 in data]
y_data=[y_row[2] for y_row in data]

#기울기의 범위는 0~10사이 t절편은 0~100사이
a1=tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
b=tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))
a2=tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
y=a1*x1 + a2*x2 +b
rmse=tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
learning_rate = 0.1
gradient_decent=tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse) 
#학습이 진행되는 부분
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(2001):
    sess.run(gradient_decent)
    if step % 100 == 0 :
      print('Epoch: %.f,RMSE=%.04f, 기울기 a1=%.4f,기울기 a2=%.4f, y절편 b=%.4f'%(step,sess.run(rmse),sess.run(a1),sess.run(a2),sess.run(b)))
      z=sess.run(y)
      print('r1=%d...r2=%d...r3=%d...r4=%d'%(z[0],z[1],z[2],z[3]))

#단순선형회기
ys=[2.3*i[0]+79  for i in data ]
print(ys)
ys=np.array(ys)
print('단순 선형회귀의 점수 평균 :',ys.mean())
results=[abs((2.3*i[0]+79)-i[2])  for i in data ]
results=np.array(results)
print('단순 선형회귀의 오차 평균 :',results.mean())

#numpy 없이
ma1=1.2301
ma2=2.1633
mb=77.8117
mresulty=[]
for i in range(4):
  mresulty.append(ma1*x1[i]+ma2*x2[i]+mb)
mavr=sum(mresulty)/4
print('다중 선형회귀의 점수 평균 :',mavr)
mdiffy=[]
for i in range(4):
  mdiffy.append(abs(y_data[i]-mresulty[i]))
avrd1=sum(mdiffy)/4
print('다중 선형회귀의 오차 평균 :',avrd1)

sa1=2.3
sb=79
sresulty=[]
for i in range(4):
  sresulty.append(sa1*x1[i]+sb)
savar2=sum(sresulty)/4
print('단순 선형회귀의 점수 평균 :',savar2)

sdiffy=[]
for i in range(4):
  sdiffy.append(abs(y_data[i]-sresulty[i]))
avrd2=sum(sdiffy)/4

#넘파이로 푸는 정답
import numpy as np
data=[[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
x1=np.array([x_row1[0] for x_row1 in data], dtype='f')
x2=np.array([x_row2[1] for x_row2 in data], dtype='f')
y=np.array([y_row[2] for y_row in data], dtype='f')
m_a1=1.2301
m_a2=2.1633
m_b=77.8117
m_y2=m_a1* x1 +m_a2*x2 +m_b
print('다중 선형회귀의 점수 평균 :',m_y2.mean())
m_diff_y = abs(y-m_y2)
print('다중 선형회귀의 오차 평균 :',m_diff_y.mean())

s_a1=2.3
s_b=79
s_y1=s_a1*x1 + s_b
print('단순 선형회귀의 점수 평균 :',s_y1.mean())
s_diff_y =abs(y-s_y1)
print('단순 선형회귀의 오차 평균 :',s_diff_y.mean())