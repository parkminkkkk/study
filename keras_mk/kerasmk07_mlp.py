import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x1 = np.array([[1,2], [3,4], [5,6]])
x2 = np.array([[[1,2,3], [4,5,6], [7,8,9]]])
y1 = np.array([11,12,13])
y2 = np.array([11,12,13])


print(x1.shape) #(3, 2)
print(x2.shape) 
print(y1.shape) #(3, )
print(y2.shape) 

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1, y1, epochs=100, batch_size=3)
model.fit(x2, y2, epochs=100, batch_size=3)

#4. 평가, 예측
loss = model. evaluate(x1,y1)
loss = model. evaluate(x2,y2)
print("loss :", loss)

result = model. predict([[5,6]])
result = model. predict([[7,8,9]])
print("[5,6]의 예측값 : ", result)
print("[7,8,9]의 예측값 : ", result)


