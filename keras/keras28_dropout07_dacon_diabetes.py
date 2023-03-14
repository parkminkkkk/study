#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
print(train_csv)  
# [652 rows x 9 columns] #(652,9)

test_csv= pd.read_csv(path+'test.csv', index_col=0)
print(test_csv) 
#(116,8) #outcome제외

x = train_csv.drop(['Outcome'], axis=1)
# print(x)
y = train_csv['Outcome']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, test_size=0.2,
    stratify=y
    )

#data scaling(스케일링)
scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성 (함수형모델) 
'''
input1 = Input(shape=(8, ))
dense1 = Dense(8, activation='linear')(input1)
dense2 = Dense(4, activation='relu')(dense1)
drop1 = Dropout(0.1)(dense2)
dense3 = Dense(8, activation='relu')(drop1)
dense4 = Dense(4, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)
'''

model = Sequential()
model.add(Dense(8, input_shape=(8,)))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc','mse'])

#시간저장
import datetime 
date = datetime.datetime.now()  #현재시간 데이터에 넣어줌
print(date)  #2023-03-14 11:15:21.154501
date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute
#시간을 문자데이터로 바꿈 : 문자로 바꿔야 파일명에 넣을 수 있음 
print(date) #0314_1115

#경로명 
filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04 : 4번째자리, .4: 소수점자리 - hist에서 가져옴 



from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=200, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=10000, validation_split=0.1, batch_size=32, verbose=1,
          callbacks=(es)) #[mcp])



#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('results:', results) 

y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)


#submission.csv생성
y_submit = np.round(model.predict(test_csv))  #np.round y_submit에도 해야함!!****
# print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0314_1530_dropout.csv') # 파일생성

'''
10.* RobustScaler
Epoch 00423: early stopping
results: [0.5853256583213806, 0.7633587718009949, 0.17271968722343445]
acc:  0.7633587786259542

11. *dropout, RobustScaler
Epoch 00657: early stopping
results: [0.5547052025794983, 0.7251908183097839, 0.18755170702934265]
acc:  0.7251908396946565
'''