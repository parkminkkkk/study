#실무/ 실제 데이터 대입 *통으로 외우기*
#따릉이문제 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #mse
import pandas as pd

#1. 데이터 
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
#  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count
print(train_csv.shape)  
#(1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
#  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5
print(test_csv.shape) 
#(715, 9) *count 제외된 데이터 

 #print : 에러 발생시 내가 프레임을 잘 적용시킨것이 맞는지 확인하기 위해 확인용! -> 이후 결과도 적어두기
print(train_csv.columns)    #칼럼 확인가능 
print(train_csv.info())     #결측치 확인 가능
print(train_csv.describe()) #각 칼럼별로 count(데이터개수), mean(평균), std(표준편차), min(최소값), 25%,50%,75%, max(최대값)
print(type(train_csv))      #type : pandas 위에 pandad가 잘 적용이 되었구나를 확인

 #결측치 처리 먼저 *선 결측치 후 데이터분리*
print(train_csv.isnull().sum()) # 각 컬럼별로 결측치 개수 알수 있음 
'''
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
'''
train_csv = train_csv.dropna()  #결측치제거 
print(train_csv.isnull().sum()) #결측치 0
'''
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0
'''
print(train_csv.info())
print(train_csv.shape)          #(1328, 10)

 #데이터 분리(train_set) *가장 중요*
x = train_csv.drop(['count'], axis=1)  #
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=1234
    )
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


