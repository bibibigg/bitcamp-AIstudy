import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# 1. 데이터 
datasets = pd.read_csv('C:\\ai_study\\_data\\train.csv')

# print(datasets.head(10))                    # nan값 확인 ID : ETQCZFEJ 의 Credit_Product 값이 nan -> No 변경


# print(datasets.columns)                     # 컬럼 확인
# ID Gender Age Region_Code Occupation Channel_Code Vintage Credit_Product Avg_Account_Balance Is_Active Is_Lead

x = datasets[['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product',
              'Avg_Account_Balance', 'Is_Active' ]]                                  # 독립변수 (원인)
y = datasets[['Is_Lead']]                                                           # 종속변수 (결과) target

cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product', 'Is_Active']
for col in cat_cols: 
    if x[col].dtype == 'O': # dtype이 'O'는 문자열을 의미합니다.
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
print(x.info())
       # object 타입을 변경


print(x, y)
print(x.shape, y.shape)                     # (245725, 9) (245725, 1)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

#stratifiedkfold 분류모델에서는 일반적으로 stratifiedKFold사용
n_splits = 50
random_state = 42
kfold = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle=True)

#Scaler 적용
# scaler = StandardScaler()
# scaler  = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape)         # (196580, 9) (196580, 1)
# print(x_test.shape, y_test.shape)           # (49145, 9) (49145, 1)

# 2 모델

cat = CatBoostClassifier() 
xgb = XGBClassifier()
lgbm = LGBMClassifier()
model = VotingClassifier(
    estimators=[('cat', cat), ('xgb', xgb), ('lgbm', lgbm), ],
    voting='hard', n_jobs=-1
)

# 3 훈련
model.fit(x_train, y_train)

# print('최적의 파라미터 : ', model.best_params_)           
# print('최적의 매개변수 : ', model.best_estimator_)         
# print('best_score : ', model.best_score_)                  
# print('model_score : ', model.score(x_test, y_test))     

# 4 평가, 예측

classfiers = [cat, xgb, lgbm]
for model in classfiers :
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))


# result = model.score(x_test, y_test)
# print('결과 acc : ', result)



# score = cross_val_score(model, x_train,
#                         y_train, cv=kfold) #cv = cross_validation

# y_predict = cross_val_predict(model,
#                               x_test, y_test,
#                               cv=kfold)
# acc = accuracy_score(y_test, y_predict)
# print('cv pred acc : ', acc)


# ############ feature importance ################
# print(model, " : ", model.feature_importances_)

# #시각화
# import matplotlib.pyplot as plt
# n_features = x.shape[1] #'x'를 사용하여 변수 'x'의 shape에서 feature의 개수를 가져와야 합니다.
#                             #따라서 n_features = x.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), x.columns) #x'를 사용하여 변수 'x'의 열 이름을 가져와야 합니다.
#                                                 # 따라서 plt.yticks(np.arange(n_features), x.columns)
# plt.title('credit Feature Importances')
# plt.ylabel('Feature')
# plt.xlabel('Importances')
# plt.ylim(-1, n_features)

# plt.show()


# ===============================================================================

# model = XGBClassifier()       적용

# scaler = StandardScaler()
        # 결과 acc :  0.7868348763862041
        # ALL 결과 acc :  0.7973140706073863
        # cv pred acc :  0.7819920642995218
        # ALL cv pred acc :  0.7934886560179062
        # [0.08347937 0.01780494 0.29436138 0.4473353  0.14020717 0.01681186]
        # ALL [0.01364775 0.09166227 0.01299173 0.17259517 0.2579861  0.11392023 0.21164966 0.01295119 0.11259582]
# scaler  = MinMaxScaler()
        # 결과 acc :  0.7868348763862041
        # ALL 결과 acc :  0.7973140706073863
        # cv pred acc :  0.7819717163495777
        # ALL cv pred acc :  0.7934683080679622
        # [0.08347937 0.01780494 0.29436138 0.4473353  0.14020717 0.01681186]
        # ALL [0.01364775 0.09166227 0.01299173 0.17259517 0.2579861  0.11392023 0.21164966 0.01295119 0.11259582]
# scaler = MaxAbsScaler()
        # 결과 acc :  0.7868348763862041
        # ALL 결과 acc :  0.7973140706073863
        # cv pred acc :  0.7819717163495777
        # ALL cv pred acc :  0.7934683080679622
        # [0.08347937 0.01780494 0.29436138 0.4473353  0.14020717 0.01681186]
        # ALL [0.01364775 0.09166227 0.01299173 0.17259517 0.2579861  0.11392023 0.21164966 0.01295119 0.11259582]
# scaler = RobustScaler()
        # 결과 acc :  0.786794180486316
        # ALL 결과 acc :  0.7973140706073863
        # cv pred acc :  0.7819513683996338
        # ALL cv pred acc :  0.7934683080679622
        # [0.08347937 0.01780494 0.29436138 0.4473353  0.14020717 0.01681186]
        # ALL [0.01364775 0.09166227 0.01299173 0.17259517 0.2579861  0.11392023 0.21164966 0.01295119 0.11259582]
# ===============================================================================
# model = LGBMClassifier()

# scaler = StandardScaler()
        # 결과 acc :  0.7886051480313359
        # ALL 결과 acc :  0.798616339403805
        # cv pred acc :  0.7873842710346932
        # ALL cv pred acc :  0.796215281310408
        # [649 441 355 231 644 680]
        # ALL [ 69 593 326 319 202 612 138 570 171]
# scaler  = MinMaxScaler()
        # 결과 acc :  0.7885441041815037
        # ALL 결과 acc :  0.7992064299521823
        # cv pred acc :  0.7870180079357004
        # ALL cv pred acc :  0.7973344185573303
        # [628 438 370 206 634 724]
        # ALL [ 69 581 339 330 192 605 144 569 171]
# scaler = MaxAbsScaler()
        # 결과 acc :  0.7885441041815037
        # ALL 결과 acc :  0.7992064299521823
        # cv pred acc :  0.7876080984840778
        # ALL cv pred acc :  0.7974565062569946
        # [628 438 370 206 634 724]
        # ALL [ 69 581 339 330 192 605 144 569 171]
# scaler = RobustScaler()
        # 결과 acc :  0.788177841082511
        # ALL 결과 acc :  0.7992267779021264
        # cv pred acc :  0.7880557533828467
        # ALL cv pred acc :  0.7966222403092889
        # [662 427 360 201 637 713]
        # ALL [ 75 605 322 331 184 594 144 569 176]

# ===============================================================================

# model = CatBoostClassifier()

# scaler = StandardScaler()
        # cv pred acc :  0.7861633940380507
        # ALL cv pred acc :  0.7955844948621427
        # [16.8330311   5.35692228 43.05462708 10.7148539  16.61185503  7.4287106 ]
        # ALL [ 0.85405363 16.87483466  4.3375499  36.6884011   7.9720189  14.85546772 8.58353241  6.46532317  3.36881851]
# scaler  = MinMaxScaler()
        # cv pred acc :  0.7861633940380507
        # ALL cv pred acc :  0.7955844948621427
        # [16.8330311   5.35692228 43.05462708 10.7148539  16.61185503  7.4287106 ]
        # ALL [ 0.85405363 16.87483466  4.3375499  36.6884011   7.9720189  14.85546772 8.58353241  6.46532317  3.36881851]
# scaler = MaxAbsScaler()
        # cv pred acc :  0.7861633940380507
        # ALL cv pred acc :  0.7955844948621427
        # [16.8330311   5.35692228 43.05462708 10.7148539  16.61185503  7.4287106 ]
        # ALL [ 0.85405363 16.87483466  4.3375499  36.6884011   7.9720189  14.85546772 8.58353241  6.46532317  3.36881851]
# scaler = RobustScaler()
        # cv pred acc :  0.7861633940380507
        # ALL cv pred acc :  0.7955844948621427
        # [16.8330311   5.35692228 43.05462708 10.7148539  16.61185503  7.4287106 ]
        # ALL [ 0.85405363 16.87483466  4.3375499  36.6884011   7.9720189  14.85546772 8.58353241  6.46532317  3.36881851]

# =================================================
# voting - hard StandardScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7986
# =================================================

# =================================================

# voting - soft StandardScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7986
# =================================================

# =================================================
# voting - hard MinMaxScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7992
# =================================================

# =================================================
# voting - soft MinMaxScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7992
# =================================================

# =================================================
# voting - hard MaxAbsScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7992
# =================================================

# =================================================
# voting - soft MaxAbsScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7992
# =================================================

# =================================================
# voting - hard RobustScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7992
# =================================================

# =================================================
# voting - soft RobustScaler
# CatBoostClassifier 정확도 :  0.7984
# XGBClassifier 정확도 :  0.7973
# LGBMClassifier 정확도 :  0.7992
# =================================================