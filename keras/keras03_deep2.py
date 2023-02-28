#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=50)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model. predict([4])
print("[4]의 예측값 : ", result)


#[4]의 예측값 :  [[3.9854996]] : mae, epochs=200
#[4]의 예측값 :  [[3.9999974]] : mse, epochs=1000 add(Dense(3,9,7,5,4,3,1))
#loss :  0.002088489942252636/ [4]의 예측값 :  [[3.923916]] : mse, epochs=50 