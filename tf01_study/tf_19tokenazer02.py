from keras.preprocessing.text import Tokenizer


text1 = '나는 진짜 매우 매우 매우 매우 맛있는 밥을 엄청 많이 많이 많이 많이 먹었다.'
text2 = '나는 딥러닝이 정말 재미있다. 재미있어 하는 내가 너무 너무 너무 너무 멋있다 또 또 또 얘기해봐'

token = Tokenizer()
token.fit_on_texts([text1, text2]) # fit on을 하면서 index 생성 
# index = token.word_index 아래와 동일
print(token.word_index)
# {'매우': 1, '많이': 2, '너무': 3, '또': 4, '나는': 5, '진짜': 6,
#  '맛있는': 7, '밥을': 8, '엄청': 9, '먹었다': 10, '딥러닝이': 11,
#  '정말': 12, '재미있다': 13, '재미있어': 14, '하는': 15, '내가': 16, '멋있다': 17, '얘기해봐': 18}

# {'매우': 1, '많이': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
x = token.texts_to_sequences([text1, text2]) # text의 인덱스번호를 x에 저장
print(x)
# [5, 6, 1, 1, 1, 1, 7, 8, 9, 2, 2, 2, 2, 10]# text1
# [5, 11, 12, 13, 14, 15, 16, 3, 3, 3, 3, 17, 4, 4, 4, 18] #text2
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

x_new = x[0] + x[1] #x에 저장해두었던 text1, text2의 인덱스 번호를 x_new로 합침
print(x_new) 
# [5, 6, 1, 1, 1, 1, 7, 8, 9, 2, 2, 2, 2, 10, 5, 11, 12, 13, 14, 15, 16, 3, 3, 3, 3, 17, 4, 4, 4, 18]

# x = to_categorical([x_new]) # onehotencoding 하면 index수 +1로 만들어짐
# print(x)
# print(x.shape) # (1, 14, 9) 3차원으로 만들어짐 


######## OneHotEncoder 수정 ##########
import numpy as np
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
# x= x.reshape(-1, 11, 9)
x = np.array(x_new) # x_new의 배열을 x에 저장
print(x.shape) # (30,)
print(x.shape) # (30,)
x = x.reshape(-1, 1) 
print(x.shape) #(30, 1)

onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape) # (30, 18)

