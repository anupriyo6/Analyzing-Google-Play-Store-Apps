# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
play=pd.read_csv('../input/googleplaystore.csv')
play.head()
play.columns
play['Content Rating']
play.Rating
play.isnull().sum()
play.dropna(inplace=True)
play.shape
play.dtypes
#cleaning category to integers
categorystring=play['Category']
categoryval=play['Category'].unique()
categoryvalcount=len(categoryval)
categoryvalcount
category_dict={}
for i in range(0,categoryvalcount):
    category_dict[categoryval[i]]=i
play['category_c']=play['Category'].map(category_dict).astype(int) 
play['category_c']
play['Size'].unique()
def change_size(size):
    if 'M' in size:
        x=size[:-1]#everything except last item in the array
        y=float(x)*1000000
        return y
    if 'k' in size:
        x=size[:-1]
        y=float(x)*1000
        return y
play['Size']=play['Size'].apply(change_size)
play['Size']
play.Size.fillna(method = 'ffill', inplace = True)
play.Installs.unique()
play['Installs']=[int(i[:-1].replace(',','')) for i in play['Installs']]
play['Type'].unique()
def type_determine(x):
    if(x=='Free'):
        return 0
    if(x=='Paid'):
        return 1
play['Type']=play['Type'].apply(type_determine)
play
play['Content Rating'].unique()
content_dict={"Everyone":0,"Teen":1,"Everyone 10+":2,"Mature 17+":3,"Adults only 18+":4,"Unrated":5}
play['Content Rating']=play['Content Rating'].map(content_dict)
play.dtypes
play=play.drop(["Android Ver","Current Ver","Last Updated","App","Category"],axis=1)
play.dtypes
gnr=play['Genres'].unique()
genre={}
len(play['Genres'])
for i in range(0,len(gnr)):
    genre[gnr[i]]=i
play['Genres']=play['Genres'].map(genre).astype(int)  
play['Genres']
play['Price'].unique()
def clean_price(x):
    if x=='0':
        return 0
    else:
        x=x[1:]#item start through rest of the array
        x=float(x)
        return x
play['Price']=play['Price'].apply(clean_price)
play.dtypes
play['Reviews'].unique()
play['Reviews']=play['Reviews'].astype(int)
play['Rating'].unique()   
y=play['Rating']
x=play.drop(['Rating'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
regr=LinearRegression()
regr.fit(x_train,y_train)
pred=regr.predict(x_test)
pred[1]
y_test[1]
from sklearn.metrics import mean_squared_error
mean_squared_error(pred,y_test)
import statsmodels.api as sm
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.