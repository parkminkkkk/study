#[실습1] x1개 y2개 ensemble
#[실습2] concantenate(함수), Concantenate(클래스) 차이

#1. 데이터 

import numpy as np
x1_datasets = np.array([range(100), range(301,401)])    # 삼성, 아모레 
print(x1_datasets.shape) #(2, 100)

x1 = np.transpose(x1_datasets)
print(x1.shape) #(100, 2)


#마지막 아웃풋 노드 1개짜리가 2개 나와야함
y1 = np.array(range(2001,2101))  # 환율
y2 = np.array(range(1001,1101))  # 금리


###random_state맞춰줘야 동일하게 잘림### 
###'\' = 한번 잘라주기 (파이썬)###
from sklearn.model_selection import train_test_split
x1_train, x1_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size=0.7, random_state=640874)

#train_test_split 5개도 가능
print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)   #(70,) (30,)
print(y2_train.shape, y2_test.shape)   #(70,) (30,)

#2. 모델구성 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#======X모델 구성============================================
#2-1. 모델1.
input1 = Input(shape=(2,))
dense01 = Dense(16, activation='relu', name='stock1')(input1)
dense02 = Dense(24, activation='relu', name='stock2')(dense01)
dense03 = Dense(32, activation='relu', name='stock3')(dense02)
output1 = Dense(16, activation='relu', name='output1')(dense03)  

#======concatenate============================================
#2-4 모델 합침
from tensorflow.keras.layers import concatenate, Concatenate  #소문자(함수) #대문자(클래스) 
# merge1 = concatenate([output1], name='mg1')  # 두 모델의 아웃풋을 합병한다(concatenate모델의 input은 1,2모델의 아웃풋을 합친 것). #2개 이상은 list형태로 받음
merge1 = Concatenate()([output1])
merge2 = Dense(24, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
hidden_output = Dense(16, name='hidden1')(merge3)

'''
# def concatenate(inputs: Any, axis: int = -1) : 소문자(함수)
-concatenate는 함수로, 여러 레이어의 출력을 연결하여 하나의 텐서로 만듭니다. 
-함수의 입력으로는 연결하려는 레이어의 출력들을 리스트나 튜플 형태로 전달하며, axis 인자를 통해 연결할 축을 지정할 수 있습니다.
-concatenate는 함수형 프로그래밍 방식을 사용하며, 입력값으로 레이어를 전달합니다.

# class Concatenate(axis: int = -1) : 대문자(클래스)
-Concatenate는 클래스로, concatenate 함수와 동일한 기능을 수행합니다. 
-객체를 생성한 후 객체의 입력으로 연결할 레이어의 출력을 전달하고, axis 인자를 설정하여 연결할 축을 지정합니다.
-Concatenate는 객체 지향 프로그래밍 방식을 사용하며, 인스턴스화를 통해 레이어를 연결합니다. 
'''

#======Y모델 구성(분기점)===============================================
#2-5. 분기점 모델1
bungi1 = Dense(10, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, activation='selu', name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)


#2-6. 분기점 모델2
bungi11 = Dense(10, name='bg11')(hidden_output)
bungi12 = Dense(10, name='bg12')(bungi11)
bungi13 = Dense(10, name='bg13')(bungi12)
last_output2 = Dense(1, name='last2')(bungi13)


#2-7 모델 정의 
model = Model(inputs=input1, outputs=[last_output1, last_output2])

# model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x1_train,[y1_train,y2_train], epochs=500, batch_size=16, validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측 

loss = model.evaluate(x1_test, [y1_test, y2_test])
print("loss합:", loss[0])
print("last1_loss:", loss[1])
print("last2_loss:", loss[2])


y_predict = model.predict(x1_test)
# print("predict", y_predict)
# print(len(y_predict), len(y_predict[0]))  #2,30 #y가 몇개인지, y의 행 개수 
##list형태는 shape안됨(파이썬) // np.numpy(y_predict)해주면 shape가능


#r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print("r2_1:", r2_1)
print("r2_2:", r2_2)
print("r2스코어", (r2_1+r2_2)/2)


#rmse
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse1 = RMSE(y1_test, y_predict[0])
rmse2 = RMSE(y2_test, y_predict[1])
print("rmse:", rmse1)
print("rmse:", rmse2)


'''
1. Concatenate(클래스)
loss합: 96.75934600830078
last1_loss: 18.118839263916016
last2_loss: 78.6405029296875
r2_1: 0.9797803930450983
r2_2: 0.9122403993956891
r2스코어 0.9460103962203937
rmse: 4.256593426229218
rmse: 8.867947930920486
'''

