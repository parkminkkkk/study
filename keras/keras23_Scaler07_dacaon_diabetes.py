#이진분류 

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터 
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
print(train_csv)  
# [652 rows x 9 columns] #(652,9)

test_csv= pd.read_csv(path+'test.csv', index_col=0)
print(test_csv) 
#(116,8) #outcome제외

# print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['Outcome'], axis=1)
# print(x)
y = train_csv['Outcome']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, test_size=0.2,
    stratify=y
)

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(12, activation='linear', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #이진분류 - 마지막 아웃풋레이어, 'sigmoid'사용 (0,1사이로 한정) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',   #이진분류 - loss, 'binary_crossentropy'사용  
              metrics=['acc','mse'] 
              ) 
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
              verbose=1,
              restore_best_weights=True
             )  

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.1,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results) 
 
y_predict = np.round(model.predict(x_test)) #np.round:반올림 / 예측값을 반올림해서 0,1의 값이 나올 수 있도록해줌 (0.5까지는 0으로, 0.6부터는 1로)

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

#submission.csv생성
y_submit = np.round(model.predict(test_csv))  #np.round y_submit에도 해야함!!****
# print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0313_1040_mmScaler.csv') # 파일생성


'''
3. 0.75점*
acc:0.73 
4. 0.78점**
acc:  0.72 /seed:321847
5. [1410]
results: [0.6158730387687683, 0.7022900581359863, 0.2031799852848053]
acc:  0.7022900763358778
6. [1420]/ 0.81점***2등
*stratify=y / seed:640, Dense(8,4,8,4,1), batch_size=32
results: [0.7886713743209839, 0.7175572514533997, 0.19658119976520538]
acc:  0.71

7. *MinMaxScaler
Epoch 00348: early stopping
results: [0.5402162671089172, 0.7251908183097839, 0.17987757921218872]
acc:  0.7251908396946565

'''
