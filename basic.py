# 회귀 분석: 변수간의 함수적 관련성을 규명하기 위해 어떤 수학적 모형을 가정하고 
# 이 모형을 측정된 변수의 데이터로부터 추정하는 통계적인 분석 방법

# 선형 회귀: 독립변수와 종속변수의 관계에 대한 답을 줄수있음 , 가장 훌륭한 예측선 긋기
# 선형 회귀의 분석 조건
# 1.독립 변수값에 해당하는 종속 변수 값들은 정규분포를 이뤄야 하고 모든 정규 분포의 분산은 동일해야함
# 2.종속 변수의 값들은 모두 통계적으로 서로 독립적이어야함
# 3.다중 회귀분석의 경우 독립변수끼리 다중공선성(교집합이 매우큰경우)이 존재하지 않아야 함

# 최소 제곱법-실습 문제
# 최소제곱법은 여러개의 입력값(x)이 있는 경우,
# 결과가 여러 원인에 의해 영향을 받는경우 한계가 있음
# 구하고자 하는 기울기의 벡터(행렬)을 역행렬로 구해야하는데 무한대,분능 에서는 구할수없음
# outlier가 있으면 적용하기 힘듬

# import numpy as np
# x=[2,4,6,8]
# y=[81,93,91,97]
# #np로 평균구하기 array_like 류의 자료는 들어갈수있음 튜플 리스트 등등
# mx=np.mean(x)
# my=np.mean(y)

# divisor = sum([(mx-i)**2 for i in x])
# dividend=sum((x[i] - mx)*(y[i]-my) for i in range(len(x)))


# print('분모',divisor)
# print('분자',dividend)

# a=dividend/divisor
# b=my-(mx*a)
# print('기울기 a =',a)
# print('y절편 b =',b)

# 평균 제곱 오차 (Mean sQuared Error :MSE)
# 오차의 합을 n으로 나누면 오차합의 평균을 구할수있음
# 평균 제곱 근 오차 (Root Mean Squared Error: RMSE)
# 임의의 선을 그려서 평가하여 조금씩 수정해가는 방법을 사용
# 평균 제곱 오차에 제곱근을 씌워서 숫자를 작게 만들어서 연산속도를 빠르게하고
# 보기편하게 역으로 제곱근 값이 너무작아 변별력이 없으면 제곱하여 볼수도있음
# import numpy as np
# x=[2,4,6,8]
# y=[81,93,91,97]
# mx=np.mean(x) #x평균
# my=np.mean(y) #y평균
# #문제 y=3x+76일때 오차구하기
# #MSE = {(실제값-예측값)**2의 평균값}
# mse=sum([((3*x[i]+76) - y[i])**2 for i in range(len(x))])/len(x)

# rmse=np.sqrt(mse)
# #RMSE = np.sqrt(mse) mse의 제곱근 값
# print('RMSE값 : ',rmse)

# import numpy as np
# x=[1,2,3,4,5,6,7,8]
# y=[0.2,0.3,0.5,0.6,0.9,0.95,1.1,1.5]
# mx=np.mean(x) #x평균
# my=np.mean(y) #y평균
# #문제 y=0.1756x-0.03392 일때 오차구하기
# #MSE = {(실제값-예측값)**2의 평균값}
# def mse(x,y):
#     gap=((0.1756*x-0.03392) - y)
#     return gap**2

# MSE = sum([mse(x[i],y[i]) for i in range(len(x))])/len(x)

# rmse=np.sqrt(MSE)
# #RMSE = np.sqrt(mse) mse의 제곱근 값
# print('RMSE값 : ',rmse)

#연습문제 경사하강법-텐서플로우 미사용
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt

# data=[[2,81],[4,93],[6,91],[8,97]]
# x=[i[0] for i in data]
# y=[i[1] for i in data]

# plt.figure(figsize=(8,5)) #figre 기름 그릴 영역을 나타내는 객체를 만들어주고 편집할수있게해줌
# plt.scatter(x,y) #연속형 변수의 산점도 그래프를 표현하는 함수
# plt.show()#메모리상에 정리된 차트를 실제 화면에 보여줌

# x_data=np.array(x) #x,y가 리스트라면 정수인 b와 연산이 불가함으로 array
# y_data=np.array(y)

# a=0
# b=0

# lr=0.01 #학습률 정하기

# #몇번 반복될지 설정(0부터 세므로 원하는 반복횟수에 +1)
# epochs=2001

# #경사 하강법 시작
# for i in range(epochs):
    
#     y_pred = a*x_data +b #y를 구하는 식 세우기
#     error = y_data - y_pred #오차를 구하는 식
#     #오차 함수를 a 로 미분한값
#     a_diff = -(1/len(x_data))*sum(x_data*(error))
#     #오차 함수를 b 로 미분한값
#     b_diff=- -(1/len(x_data))* sum(error) 
#     #공식대로 코드를 짰을경우
#     a_diff = 2/len(x_data)*sum((a*x_data+b-y_data)*x_data)
#     b_diff= 2/len(x_data)*sum((a*x_data+b-y_data))

#     a = a-lr*a_diff #학습률을 곱해 기존의 a값 업데이트
#     b = b-lr*b_diff #학습률을 곱해 기존의 b값 업데이트
    
#     if i%100 ==0: #100번 반복될 때마다 현재의 a,b값 출력
#         print('epoch=%.f, 기울기=%.04f, 절편==%.04f'%(i,a,b))

# #앞서 구한 기울기와 절편을 이용해 그래프를 다시그리기    
# y_pred=a*x_data +b
# plt.scatter(x,y)
# plt.plot([min(x_data),max(x_data)],[min(y_pred),max(y_pred)])
# plt.show()

# matplotlib 사용법 익히기
# import numpy as np
# import matplotlib.pyplot as plt

# a=np.random.rand(1000)
# b=np.random.rand(10000)
# c=np.random.rand(100000)

# plt.hist(a,bins=100, density=True , alpha=0.5, histtype='step')
# plt.hist(b,bins=100,density=True,alpha=0.75,histtype='step')
# plt.hist(c,bins=100,density=True,alpha=1.0,histtype='step')
# plt.show()

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

# x=np.linspace(-np.pi,np.pi,256)
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