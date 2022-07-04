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