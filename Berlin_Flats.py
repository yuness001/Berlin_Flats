
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

X=df[['Rooms','Space']]
y=df['Rent']

plt.scatter(X['Space'],y,color='red',label='space')
#plt.legend()
plt.title('relation between rent and space')
plt.scatter(X['Rooms'],y,color='green',label='rooms')
plt.legend()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LinearRegression()

model.fit(X_train,y_train)

score=model.score(X_test,y_test)

print('the score of the primary model is :',str(score)[:4],'%')

