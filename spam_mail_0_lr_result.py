import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

random.seed(42)


path = './_data/project/'
save_path = '/_save/project/'

# Load Data
dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
dt_kor = pd.read_csv(path + 'kor_spam_ham_dataset3.csv')

# Preprocess the data
dt_eng.drop('Unnamed: 0', axis=1, inplace= True)
dt_eng.columns = ['label', 'text', 'class']
dt_eng.head()
#Eng_Text processing 
#Remove stopwords from the data
stopwords = set(stopwords.words('english'))
dt_eng['text'] = dt_eng['text'].apply(lambda x: ' '.join([ word for word in word_tokenize(x)
                                                          if not word in stopwords]))

# 먼저 train 데이터와 test 데이터 인덱스 없이 배열로 만들기
kor_x = np.array([x for x in dt_kor['text']])
# print(kor_x)
kor_y = dt_kor.loc[:,'class']

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']

y=np.array(dt_kor['class'])
_, y_indices = np.unique(y, return_inverse=True)
print(np.bincount(y_indices))
# Split the Korean data into training and testing sets
X_korean_train, X_korean_test, y_korean_train, y_korean_test = train_test_split(
    dt_kor['text'], y, test_size=0.3, random_state=42,stratify=y)

# Split the English data into training and testing sets
X_english_train, X_english_test, y_english_train, y_english_test = train_test_split(
    dt_eng['text'], dt_eng['class'], test_size=0.3, random_state=42,stratify=dt_eng['class'])

# Feature extraction
vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
X_korean_train_features = vectorizer.fit_transform(X_korean_train).toarray()
X_korean_test_features = vectorizer.transform(X_korean_test).toarray()
X_english_train_features = vectorizer.fit_transform(X_english_train).toarray()
X_english_test_features = vectorizer.transform(X_english_test).toarray()
print(X_korean_train_features.shape[0], X_korean_train_features.shape[1])  
print(X_english_train_features.shape[0], X_english_train_features.shape[1]) 

# 변환된 시퀀스 번호를 이용해 단어 임베딩 벡터 생성
max_length = 230
padding_type='pre'
train_korx = pad_sequences(X_korean_train_features, padding='pre', maxlen=max_length)
test_korx = pad_sequences(X_korean_test_features, padding=padding_type, maxlen=max_length)

print(train_korx.shape, test_korx.shape) #(90, 1000) (39, 1000)
# print(train_korx)

train_engx = pad_sequences(X_english_train_features, padding='pre', maxlen=max_length)
test_engx = pad_sequences(X_english_test_features, padding=padding_type, maxlen=max_length)
print(train_engx.shape, test_engx.shape) #(3619, 1000) (1552, 1000)

# Train the Korean logistic regression model
korean_clf = LogisticRegression(random_state=42).fit(train_korx, y_korean_train)

# Train the English logistic regression model
english_clf = LogisticRegression(random_state=42).fit(train_engx, y_english_train)

# Create a voting classifier with the two models
E_clf = GradientBoostingClassifier()
# voting_clf = VotingClassifier(estimators=[('korean', korean_clf), ('english', english_clf)], voting='soft')

# Fit the voting classifier on the training data
E_clf.fit(train_korx, y_korean_train)
# voting_clf.fit(train_korx, y_korean_train)


# Test the voting classifier on the testing data
y_pred = E_clf.predict(test_korx)

#lr
# Evaluate the performance of the model
accuracy = accuracy_score(y_korean_test, y_pred)
precision = precision_score(y_korean_test, y_pred)
recall = recall_score(y_korean_test, y_pred)
f1 = f1_score(y_korean_test, y_pred)

print('Accuracy:', accuracy) 
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Predict the label of a new email in Korean
new_email_korean = ['광고. 스팸 이메일입니다.']
new_email_korean = vectorizer.fit_transform(new_email_korean).toarray()
new_email_korean = pad_sequences(new_email_korean, padding='pre', maxlen=max_length)
# print(new_email_korean.shape) #(1, 230)
mail_pred = E_clf.predict(new_email_korean)
print('Prediction:', mail_pred)

'''
[최종]
(253, 8862) (109, 8862)
(3619, 8862) (1552, 8862)
Prediction: [1]
Accuracy: 0.963302752293578
Precision: 0.9615384615384616
Recall: 1.0
F1-score: 0.9803921568627451
'''

'''
#E_clf = GradientBoostingClassifier()
Accuracy: 0.926605504587156
Precision: 0.9259259259259259
Recall: 1.0
F1-score: 0.9615384615384615
'''
'''
#data2
vectorizer = TfidfVectorizer()
Accuracy: 0.7692307692307693
Precision: 0.7692307692307693
Recall: 1.0
F1-score: 0.8695652173913044
'''

'''
#data3
vectorizer = TfidfVectorizer()
#max_len : 1000(임의값)
Accuracy: 0.9174311926605505
Precision: 0.9174311926605505
Recall: 1.0
F1-score: 0.9569377990430622

#max_len : 230 / 8862
Accuracy: 0.9174311926605505
Precision: 0.9174311926605505
Recall: 1.0
F1-score: 0.9569377990430622
======================================
CountVectorizer()
#max_len :  8862
Accuracy: 0.9357798165137615
Precision: 0.9345794392523364
Recall: 1.0
F1-score: 0.966183574879227

#max_len : 230 
Accuracy: 0.9174311926605505
Precision: 0.9174311926605505
Recall: 1.0
F1-score: 0.9569377990430622
'''




# # Predict the label of a new email in Korean
# new_email_korean = ['스팸 이메일입니다.']
# new_email_korean_features = vectorizer.transform(new_email_korean)
# new_email_korean_pred = voting_clf.predict(new_email_korean_features)
# print('Prediction:', new_email_korean_pred)

# # Predict the label of a new email in English
# new_email_english = ['Buy cheap Viagra now!']
# new_email_english_features = vectorizer.transform(new_email_english)
# new_email_english_pred = voting_clf.predict(new_email_english_features)
# print('Prediction:', new_email_english_pred)



##############################
# #Tokenizer
# vocab_size = 2000 
# tokenizer = Tokenizer(num_words = vocab_size)  
# tokenizer.fit_on_texts(X_korean_train) 
# X_korean_train_features = tokenizer.texts_to_sequences(X_korean_train)
# print(X_korean_train_features)
# print(X_korean_train_features.__class__)
# print(X_korean_train_features[0].__class__)
# X_korean_test_features = tokenizer.texts_to_sequences(X_korean_test)  
# print(len(X_korean_train_features[0]), len(X_korean_test_features)) #69 30

# word_index = tokenizer.word_index
# max_length = 50
# padding_type='pre'
# train_korx = pad_sequences(X_korean_train_features, padding='pre', maxlen=max_length)
# test_korx = pad_sequences(X_korean_test_features, padding=padding_type, maxlen=max_length)

# print(train_korx)
# train_korx = test_korx[0]
# print(train_korx)
