import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print("Original : \n", a)
a_transpose = np.transpose(a) # transpose 행열을 바꾸는 함수 따로 행과 열을 정해주지 않음
print("Transpose : \n", a_transpose)
#  [[1 4]
#  [2 5]
#  [3 6]]

a_reshape = np.reshape(a, (3,2)) # reshape 행열을 바꾸는 함수, 행과 열을 정해줘야 함(3,2는 3행 2열로 정렬)
print("Reshape : \n", a_reshape) 
#  [[1 2]
#  [3 4]
#  [5 6]]

# transpose와 reshape의 차이점은 transpose는 각 리스트의 순서대로 묶이고 
# reshape는 리스트 처음부터 끝까지의 순서대로 묶음