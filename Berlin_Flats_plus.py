
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('https://raw.githubusercontent.com/MSadriAghdam/Berlin_Flats/main/berlin_flats.csv')

df.sample(5)

df.describe()

df.info()

df=df.drop(columns='Unnamed: 0')

df.shape

df.Region.unique()

df.Region.nunique()

smallest_space=df[df.Space == df.Space.min()]
smallest_space

largest_space=df[df.Space == df.Space.max()]
largest_space

len(df.Region.value_counts())

len(df.Condition.value_counts())

Condition=pd.get_dummies(df['Condition'])

df.sample(5)

df=df.drop(columns='Condition')

df.sample(5)

df.join(Condition)

Region=pd.get_dummies(df['Region'])

df=df.drop(columns='Region')

df=df.join(Region)

df=df.join(Condition)

df.sample(5)

correlated_df=df.corr()

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlated_df)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df[['Rent','Rooms','Space']].corr(),annot=True)

df=df[['Rent','Rooms','Space']]

df.sample(5)

#X=df[['Rooms','Space']]
X=df['Space'].values.reshape(-1,1)
y=df['Rent']

plt.scatter(X,y,color='red',label='space')
#plt.legend()
plt.title('relation between rent and space')
#plt.scatter(X,y,color='green',label='rooms')
#plt.legend()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=LinearRegression()

model.fit(X_train,y_train)

score=model.score(X_test,y_test)

print('the score of the first model is :',str(score))

df2=df[df['Space']<250]

X2=df2['Space'].values.reshape(-1,1)
y2=df2['Rent']

plt.scatter(X2,y2,color='red',label='space')

X_train2,X_test2,y_train2,y_test2=train_test_split(X2,y2,test_size=0.3)

model2=LinearRegression()

model2.fit(X_train2,y_train2)

score2=model2.score(X_test2,y_test2)

print('the score of the second model is :',str(score2))

print('the score of the first model is : {}% \nthe score of the second model is : {}%'.format (str(score)[:4],str(score2)[:4]))

df3=df2[df2['Rent']<4000]

X3=df3['Space'].values.reshape(-1,1)
y3=df3['Rent']

plt.scatter(X3,y3,color='red',label='space')

X_train3,X_test3,y_train3,y_test3=train_test_split(X3,y3,test_size=0.3)

model3=LinearRegression()

model3.fit(X_train3,y_train3)

score3=model3.score(X_test3,y_test3)

print('the score of the third model is :',str(score3))

print('the score of the first model is : {}% \nthe score of the second model is : {}% \nthe score of the third model is : {}%'.format (str(score)[:4],str(score2)[:4],str(score3)[:4]))

df4=df3[df3['Space']<180]

X4=df4['Space'].values.reshape(-1,1)
y4=df4['Rent']

plt.scatter(X4,y4,color='red',label='space')

X_train4,X_test4,y_train4,y_test4=train_test_split(X4,y4,test_size=0.3)

model4=LinearRegression()

model4.fit(X_train4,y_train4)

score4=model4.score(X_test4,y_test4)

print('the score of the fourth model is :',str(score4))

print('the score of the first model is : {}% \nthe score of the second model is : {}% \nthe score of the third model is : {}% \nthe score of the fourth model is : {}%'.format (str(score)[:4],str(score2)[:4],str(score3)[:4],str(score4)[:4]))

print(model.score(X_test,y_test))
print(model.score(X_test2,y_test2))
print(model.score(X_test3,y_test3))
print(model.score(X_test4,y_test4))

print(model2.score(X_test,y_test))
print(model2.score(X_test2,y_test2))
print(model2.score(X_test3,y_test3))
print(model2.score(X_test4,y_test4))

print(model3.score(X_test,y_test))
print(model3.score(X_test2,y_test2))
print(model3.score(X_test3,y_test3))
print(model3.score(X_test4,y_test4))

print(model4.score(X_test,y_test))
print(model4.score(X_test2,y_test2))
print(model4.score(X_test3,y_test3))
print(model4.score(X_test4,y_test4))

