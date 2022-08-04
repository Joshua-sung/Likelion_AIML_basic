# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
# a = tf.constant(7.0, name = 'data1')
# b = tf.constant(3.0, name = 'data2')
# c = tf.constant(2.0, name = 'data3')
# v = a * b / c
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(v))

# 텐서플로우는 node와 edge로 구성된 그래프로 표현, 그래프의 각 node들은 operation을 의미한다

# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
# a = tf.constant(5, name = 'data1')
# b = tf.constant(3, name = 'data2')
# c = tf.multiply(a,b,name='c')
# d=tf.add(a,b,name='d')
# e=tf.add(c,d,name='e')
# sess= tf.Session()
# print(sess.run(e))

# #행렬 예제
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()

 
# input_data=[[1.,2.,3.],[1.,2.,3.],[2.,3.,4.]]
# x=tf.placeholder(dtype=tf.float32,shape=[None,3])
# w=tf.Variable([[2.],[2.],[2.]], dtype=tf.float32)
# y=tf.matmul(x,w)
# sess=tf.Session()

# init=tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y,feed_dict={x:input_data})
# sess.close()
# print(result)

# #브로드 캐스팅 행렬 연산에서 차원이 맞지 않을떄 차원을 늘려줘서 자동으로 맞춰줌(줄이는건x)

# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()

 

# input_data=[[1,1,1],[2,2,2,]]
# x=tf.placeholder(dtype=tf.float32,shape=[2,3])
# w=tf.Variable([[2],[2],[2]], dtype=tf.float32)
# b=tf.Variable([4], dtype=tf.float32)          
# y=tf.matmul(x,w)+b

# print(x.get_shape())

# sess=tf.Session()

# init=tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y,feed_dict={x:input_data})
# sess.close()
# print(result)

# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()


# x=tf.constant([[1.0,2.0,3.0]])
# w=tf.constant([[2.0],[2.0],[2.0]])

# y=tf.matmul(x,w)
# print(x.get_shape())
# print(w.get_shape())
# print(y.get_shape())
# print(y)

 

# sess=tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y)
# sess.close()
# print(result)

# plt 연습
# import tensorflow as tf
 
# a=tf.constant(2)
# b=tf.constant(3)
# c=tf.constant(5)
 
# add=tf.add(a,b)
# sub=tf.subtract(a,b)
# mul=tf.multiply(a,b)
# div=tf.divide(a,b)

# print('add=', add.numpy())
# print('sub=', sub.numpy())
# print('mul=',mul.numpy())
# print('div=',div.numpy())
 
# mean=tf.reduce_mean([a,b,c])
# sum= tf.reduce_sum([a,b,c])
# print('mean=',mean.numpy())
# print('sum=',sum.numpy())

# matrix1=tf.constant([[1.,2.],[3.,4.]])
# matrix2=tf.constant([[5.,6.],[7.,8.]])
# product=tf.matmul(matrix1,matrix2)
# print(product)
# print(product.numpy())
 
# import numpy as np
# import matplotlib.pyplot as plt

# a=np.random.rand(1000)
# b=np.random.rand(10000)
# c=np.random.rand(100000)
 
# plt.hist(a,bins=100, density=True , alpha=0.5, histtype='step')
# plt.hist(b,bins=100,density=True,alpha=0.75,histtype='step')
# plt.hist(c,bins=100,density=True,alpha=1.0,histtype='step')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
 
# a=np.random.randint(0,10,100)
# b=np.random.randint(0,10,1000)
# c=np.random.randint(0,10,10000)
# d=np.random.randint(0,10,100000)

# plt.subplot(221)
# plt.hist(a,bins=10)
# plt.title('axes 1')
# plt.subplot(222)
# plt.hist(a,bins=10)
# plt.title('axes 2')
# plt.subplot(223)
# plt.hist(a,bins=10)
# plt.title('axes 3')
# plt.subplot(224)
# plt.hist(a,bins=10)
# plt.title('axes 4') 
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
 
# plt.style.use('default')
# plt.rcParams['figure.figsize']=(6,3)
# plt.rcParams['font.size']=12
 
# a=np.random.rand(1000)
# b=np.random.rand(10000)
# c=np.random.rand(100000)

#  plt.hist(a,bins=100, density=True , alpha=0.5, histtype='step',label='1000')
# plt.hist(b,bins=100,density=True,alpha=0.75,histtype='step',label='10000')
# plt.hist(c,bins=100,density=True,alpha=1.0,histtype='step',label='100000')

# plt.legend()
# plt.show()

# import numpy as np
# x=np.linspace(-np.pi,np.pi,256)
# c=np.cos(x)
# plt.title('x,y tick label set')
# plt.plot(x,c)
# plt.xticks([-np.pi,np.pi/2,0,np.pi/2,np.pi])
# plt.yticks([-1, 0 ,+1])
# plt.show() 

# import numpy as np
# x=np.linspace(-np.pi,np.pi,256)
# c=np.cos(x)
# plt.title('x,y tick label set')
# plt.plot(x,c)
# plt.xticks([-np.pi,np.pi/2,0,np.pi/2,np.pi],[r'$-\pi/2$',r'$0$',r'$+\pi/2$',r'$+\pi$'])
# plt.yticks([-1, 0 ,+1],['Low','Zero','High'])
# plt.show()

# import numpy as np
# x=np.linspace(-np.pi,np.pi,256)
# c=np.cos(x)
# plt.title('del grid')
# plt.plot(x,c)
# plt.xticks([-np.pi,np.pi/2,0,np.pi/2,np.pi],[r'$-\pi/2$',r'$0$',r'$+\pi/2$',r'$+\pi$'])
# plt.yticks([-1, 0 ,+1],['Low','Zero','High'])
# plt.grid(True)
# plt.show()

# t=np.arange(0.,5.,0.2)
# plt.title('asd')
# plt.plot(t,t,'r--',t,0.5*t**2,'bs:',t,0.2*t**3,'g^-')
# plt.show()

# #여러개의 선을 그릴떄
# plt.title('more than 2 together')
# plt.plot([1,4,9,16],c='b',lw=5,ls='--',marker='o',ms=15,mec='g',mew=5,mfc='r')
# plt.plot([9,16,4,1],c='k',lw=5,ls=':',marker='s',ms=10,mec='m',mew=5,mfc='c')
# plt.show()

#  x=np.linspace(-np.pi,np.pi,256)
# c,s=np.cos(x),np.sin(x)
# plt.title('mark legend')
# plt.plot(x,c,ls='--',label='cosine')
# plt.plot(x,s,ls=':',label='sine')
# plt.legend(loc=2)
# plt.show()

# #범례 표시하기
# x=np.linspace(-np.pi,np.pi,256)
# c,s=np.cos(x),np.sin(x)
# plt.plot(x,c,label='cosine')
# plt.xlabel('time')
# plt.ylabel('amplitude')
# plt.title('cosine plot')
# plt.show()

# #y,x축의 이름 라벨링
# np.random.seed(2)
# f1=plt.figure(figsize=(10,2))
# plt.title('figsize=(10,2)')
# plt.plot(np.random.randn(100))
# plt.show()

# f1=plt.figure(1)
# plt.title('current figure ')
# plt.plot([1,2,3,4],'ro:')
# f2=plt.gcf()
# print(f1,id(f1))
# print(f2,id(f2))
# plt.show()

# x1=np.linspace(0.0,5.0)
# x2=np.linspace(0.0,2.0)
# y1=np.cos(2*np.pi*x1)*np.exp(-x1)
# y2=np.cos(2*np.pi*x2)
# ax1=plt.subplot(2,1,1)
# plt.plot(x1,y1,'yo-')
# plt.title('A take of 2 subplots')
# plt.ylabel('Damped oscillation')
# print(ax1)

# ax2=plt.subplot(2,1,2)
# plt.plot(x2,y2,'r.-')
# plt.xlabel('time(s)')
# plt.ylabel('undamped ')
# print(ax2)

# plt.tight_layout()
# plt.show()
# np.random.seed(0)
# #subplot(가로,세로,사분면) (221)-> 2x2배치 1사분면에 배치
# plt.subplot(221)
# plt.plot(np.random.rand(5))
# plt.title('axes 1')
# plt.subplot(222)
# plt.plot(np.random.rand(5))
# plt.title('axes 2')
# plt.subplot(223)
# plt.plot(np.random.rand(5))
# plt.title('axes 3')
# plt.subplot(224)
# plt.plot(np.random.rand(5))
# plt.title('axes 4')
# plt.tight_layout()
# plt.show()

# # 같은 그래프그리기를 2차원 배열로 표현하기

# fig,axes=plt.subplots(2,2)
# np.random.seed(0)
# axes[0,0].plot(np.random.rand(5))
# axes[0,0].set_title('axes 1')
# axes[0,1].plot(np.random.rand(5))
# axes[0,1].set_title('axes 2')
# axes[1,0].plot(np.random.rand(5))
# axes[1,0].set_title('axes 3')
# axes[1,1].plot(np.random.rand(5))
# axes[1,1].set_title('axes 4')

# plt.tight_layout()
# plt.show()

# 같은 그래프그리기를 2차원 배열로 표현하기
fig,axes=plt.subplots(2,2)
np.random.seed(0)
axes[0,0].plot(np.random.rand(5))
axes[0,0].set_title('axes 1')
axes[0,1].plot(np.random.rand(5))
axes[0,1].set_title('axes 2')
axes[1,0].plot(np.random.rand(5))
axes[1,0].set_title('axes 3')
axes[1,1].plot(np.random.rand(5))
axes[1,1].set_title('axes 4')
plt.tight_layout()
plt.show()

#여러가지 플롯의 여러가지 종류
#bar chart
color=['blue','green','red']
y=[2,3,1]
x=np.arange(len(y))
xlabel=['A','B','C']
plt.title('barchart')
plt.bar(x,y)
plt.yticks(sorted(y))
plt.yticks(y)
plt.xlabel("ABC")
plt.ylabel('often')
plt.show()

#barh chart
np.random.seed(0)
people=['ja','na','da','la']
y_pos=np.arange(len(people))
performance=3+10*np.random.rand(len(people))
error = np.random.rand(len(people))
print(performance)
print(error)
plt.title('barh chart')
plt.barh(y_pos,performance,xerr=error,alpha=0.4)

#alpha는 투병도 1은 완전 불투명 error 바위에 겹처그리는거
plt.yticks(y_pos,people)
plt.xlabel=('x label')
plt.show() 

#stem chart
x=np.linspace(0.1,2*np.pi,10)
plt.title('stem plot')
plt.stem(x,np.cos(x),'-.')
plt.show()

#파이차트
labels=['ja','na','da','la']
sizes=[20,30,45,10]
colors=['yellowgreen','gold','lightskyblue','lightcoral']
explode=(0,0.1,0.2,0.3)

#explode 원의 중심에서 얼마나 떨어지는가
plt.title('Pie chart',fontdict={'fontsize' : 20})
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='$1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

#히스토그램
np.random.seed(0)
x=np.random.randn(1000)
plt.title('Histogram')
arrays,bins,patches=plt.hist(x,bins=10)
plt.show()