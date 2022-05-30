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

import numpy as np
x=[1,2,3,4,5,6,7,8]
y=[0.2,0.3,0.5,0.6,0.9,0.95,1.1,1.5]
mx=np.mean(x) #x평균
my=np.mean(y) #y평균
#문제 y=0.1756x-0.03392 일때 오차구하기
#MSE = {(실제값-예측값)**2의 평균값}
def mse(x,y):
    gap=((0.1756*x-0.03392) - y)
    return gap**2

MSE = sum([mse(x[i],y[i]) for i in range(len(x))])/len(x)

rmse=np.sqrt(MSE)
#RMSE = np.sqrt(mse) mse의 제곱근 값
print('RMSE값 : ',rmse)