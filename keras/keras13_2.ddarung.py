#데이콘 따릉이 문제풀이(대회) *중요* 많이 써먹음!! 머리에 다 넣기!!
#train.csv 파일 :  x,y로 구분해서 훈련시키고 
#test.csv 파일 : 그대로 predict파일에 집어넣음 (count값 없는 파일이므로..)-> count값 구함
#submission.csv 파일 : predict값(count)구한 것을 넣어서 파일 제출!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
 #r2스코어, mse 궁금함 -> 불러옴/ mse지표에 루트씌워서 rmse지표 만들어주기 
import pandas as pd 
 #깔끔하게 데이터화 됨(csv 데이터 가져올때 좋음) *실무에서 엄청 씀 pandas*

#1. 데이터 
path = './_data/ddarung/' 

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #0번째는 인덱스 칼럼이야  
print(train_csv) #확인
print(train_csv.shape) # (1459, 10)


test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0) 
print(test_csv)
print(test_csv.shape) #(715, 9) : count 없어서 (10->9)


############# <결측치 처리 먼저> ##########################################################
'''
먼저 결측치 처리하는 이유 : 통 데이터일때 결측치 처리한다. 왜냐하면 나중에 할 경우 없어질 수도 있으니까..
loss=nan나오는 문제 해결하기 위함
'''
#1. 결측치 처리 = 제거 
# print(train_csv.isnull()) : true/False
print(train_csv.isnull().sum()) #isnull이 True인것의 합계 : 각 컬럼별로 결측치 몇개인지 알수 있음
train_csv = train_csv.dropna()   ### *결측치제거* 결측치 삭제하겠다 (함수형태) -> train_csv로 명명
print(train_csv.isnull().sum()) #결측치 제거 후 확인용 
print(train_csv.info())
print(train_csv.shape)       #(1328, 10)

##########################################################################################


############# <train_csv데이터에서 x와 y를 분리> ############################################
#y데이터 분리 *가장 중요(외우기)*
#x : count컬럼 뺀 전부 / y : count컬럼
#train.csv 파일 :  x,y로 구분해서 훈련시키고 
#test.csv 파일 : 그대로 predict파일에 집어넣음 (count값 없는 파일이므로..)-> count값 구함
#submission.csv 파일 : predict값(count)구한 것을 넣어서 파일 제출!

x = train_csv.drop(['count'], axis=1) 
print(x)
#1. train_set #2. axis=1 : 열  
#3. [] : 2개이상은 list, 여러개로 다른 것도 할 수 있다
#변수지정 : x = train_csv에서 count를 drop시킨것

y = train_csv['count']  
print(y)

###########################################################################################

x_train, x_test, y_train, y_test = train_test_split(
      x, y, shuffle=True, train_size=0.8, random_state=650874
      )
                                                         #결측치 제거 후 
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9) -> (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021, ) (438,)    -> (929,) (399,)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=30, batch_size=32,
          verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

########### 현재까지, loss = mse임 / 그러나, 'rmse의 loss값' 달라고했으니까 수정해줘야함 ############# 
def RMSE(y_test, y_predict): # RMSE함수 정의 : rmse라는 함수는 ~~로 정의 할거야! / #함수 쓰는 이유 : 재사용할 때
    return np.sqrt(mean_squared_error(y_test, y_predict)) #mse에 루트(np.sqrt) 씌움
rmse = RMSE(y_test, y_predict) # RMSE함수 사용(실행)
print("RMSE : ", rmse)

########### [실습] submission.csv를 만들어보자###########################################
# print(test_csv.isnull().sum()) #nan값(결측치) 개수 확인/  #test_csv에도 결측치가 있다. 

y_submit = model.predict(test_csv)  # y=test_csv라고 위에서 명명했었음 => predict(y):y 예측값을 y_submit으로 명명하겠다!
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission) #아직 칼럼에 count값 안 넣어서 NaN이라고 뜸 
submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submit_0306_0447.csv') #파일생성
#저장하때는 to_csv / 파일 읽어올때(불러올때는) read_csv
#다시 실행시킬때마다 파일명 같은경우 파일 덮어씀 (파일명 변경(수동)-> 나중에는 자동으로 가능)


'''
*loss값 줄이기 (10개 이상)
1. loss :  3377.663330078125 / RMSE :  58.11766868637356 /  *[58*58=3377:loss]
 81점 / train_size=0.8, random_state=650874, Dense(32,16,8,4,1), mse, epochs=30, batch_size=32, verbose=1
2/ loss :  2444.00732421875 / RMSE :  49.43689952875558 
 점 / train_size=0.8, random_state=650874, Dense(32,16,8,4,1), mse, epochs=1000, batch_size=32, verbose=1

3. [0742] loss :  2412.064697265625/ r2스코어 :  0.6031572169071889/ RMSE :  49.11277445641966
73.75점 /train_size=0.8, random_state=650874, Dense(8,4,2,1), mse, epochs=3000, batch_size=50, verbose=3

'''
