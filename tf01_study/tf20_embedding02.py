import numpy as np
from keras.preprocessing.text import Tokenizer

# 1. 데이터
docs = ['재밌어요', '재미없다', '돈아깝다', '숙면했어요',
        '최고에요', '꼭봐라', '세번봐라', '또보고싶다',
        'n회차관람', '배우가잘생기긴했어요', '발연기에요', '추천해요',
        '최악', '후회된다', '돈버렸다', '글쎄요', '보다 나왔다',
        '망작이다', '연기가 어색해요', '차라리기부할걸',
        '다음편 나왔으면 좋겠다', '다른거볼걸', '감동이다']

# 긍정 1, 부정 0
labels = np.array([1, 0, 0, 0,
                   1, 1, 1, 1,
                   1, 0, 0, 1,
                   0, 0, 0, 0, 0,
                   0, 0, 0,
                   1, 0, 1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'재밌어요': 1, '재미없다': 2, '돈아깝다': 3, '숙면했어요': 4, '최고에요': 5, '꼭봐라': 6, '세번봐라': 7, '또보고싶다': 8, 'n회차관람': 9, '배우가잘생기긴했어요': 10, '발연기에요': 11, '추천해요': 12, '최악': 13, '후회된다': 14, '돈버렸다': 15, '글쎄요': 16, '보다': 17, '나왔다': 18, '망작이다': 19, '연기가': 20, '어색해요': 21, '차라리기부할걸': 22, '다음편': 23, '나왔으면': 24, '좋겠다': 25, '다른거볼걸': 26, '감동이다': 27}

x = token.texts_to_sequences(docs)
print(x)
# [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17, 18], [19], [20, 21], [22], [23, 24, 25], [26], [27]]

# pad_sequences
from keras_preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=3)
print(pad_x)
print(pad_x.shape) #(23, 3) => 3은 maxlen

word_size = len(token.word_index)
print('word_size : ', word_size) # word_size :  27

# 모델

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=10, input_length=3)) # 인풋딤에는 워드사이즈 +1, 아웃풋딤은 노드수로 이해 아무렇게나 넣을 수 있음 인풋랭스는 제일 길었던 길이
model.add(LSTM(32)) #embedding을 사용시 3사원을 받아서 2차원을 출력해줌 embedding을 사용하지 않을 경우 위에서 reshape을 해서 차원을 맞춰주어야 함
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #긍정, 부정의 이진분류라 sigmoid사용
# model.summary()

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=10, batch_size=1)

# 평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)


#################### [predict] ########################
x_predict = '영화가 정말 재미없네'

# 1) tokenizer
token = Tokenizer()
x_predict = np.array([x_predict])
print(x_predict)
token.fit_on_texts(x_predict)
x_pred = token.texts_to_sequences(x_predict)
print(token.word_index)
print(len(token.word_index))
print(x_pred)

# 2) pad_sequences
x_pred = pad_sequences(x_pred, padding='pre')
print(x_pred)
# 3) model.predict
y_pred = model.predict(x_pred)
print(y_pred)

# [실습]
# 1. predict 결과 제대로 출력되도록
# 2. score를 '긍정' '부정'으로 출력하라

score = float(model.predict(x_pred))

if y_pred < 0.5:
    print("{:.2f}% 확률로 부정\n".format((1-score) * 100))
else :
    print("{:.2f}% 확률로 긍정\n".format(score * 100))