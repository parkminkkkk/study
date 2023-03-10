
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 

#1. 데이터 
datasets = fetch_covtype()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
print('y의 라벨값 :', np.unique(y)) #y의 라벨값 : [1 2 3 4 5 6 7] 
print(y)

#1)one hot encoding 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y)
print(y.shape) #(581012, 7) 

#2)데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, train_size=0.8,stratify=y)

#2. 모델구성 
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=54))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#ES추가 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True, verbose=1)

hist = model.fit(x_train, y_train, epochs=30000, batch_size=1024, validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측 

results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1) #print(y_test.shape)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'])
plt.title('binary_crossentropy')
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()


'''
1. 
results: [0.6942906975746155, 0.7130711078643799]
acc: 0.7130710911078028
2.
results: [0.6665192246437073, 0.7229589819908142]
acc: 0.7229589597514694
'''