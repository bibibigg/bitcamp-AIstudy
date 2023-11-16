# 1. 파이썬 기초
# 리스트

a = [38, 21, 45, 98, 55]
print (a)
print (a[0])

# 리스트 문자 출력
#4번 예제
e = ['메이킷', '우진', '시은']
print(e)
print (e[0])
print (e[1])
print (e[2])

#리스트 정수와 문자 출력

c = ['james', 26, 175.3, True]
print(c)

#5번 예제
f = ['메이킷', '우진', '제임스', '시은']

print (f[:2])
print (f[1:4])
print (f[2:4])
print (f[0:4])

# extend() 함수 사용하여 리스트 이어붙이기
#6번 예제
g = ['우진', '시은']
h = ['메이킷', '소피아', '하워드']

g.extend(h) # extend를 사용하여 이어붙임 g.extend(h)는 g리스트 뒤에 h리스트를 이어서 붙임
print(g)
print(h) # h에는 이어붙이지 않아서 리스트 h만 나옴

#7번 예제
i = ['우진', '시은']
j = ['메이킷', '소피아', '하워드']

j.extend(i) 
print(i) # 리스트 i에는 extend를 붙이지 않아서 이어서붙여지지 않음
print(j)