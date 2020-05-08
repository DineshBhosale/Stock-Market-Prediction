# -*- SUPPORT VECTOR REGRESSION -*-
"""
Created on Tue Nov 26 18:31:41 2019

@author: HP
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

data=pd.read_csv('D:/NCSU/ECE 592 - Special topics in Data Science/FINAL PROJECT/googlestockdata.csv')
data=data.iloc[::-1]
data['Dates']=data.Date.str.replace("-","").astype(int)
data=data.set_index('Date')

names=data.columns
scaler=preprocessing.StandardScaler()
scaled_data=scaler.fit_transform(data)
data=pd.DataFrame(scaled_data,columns=names)
print(data.isnull().sum())

correlation_matrix = data.corr().round(2)
print(correlation_matrix)
plt.figure(2,figsize=(7,7))
sns.heatmap(data=correlation_matrix, annot=True, yticklabels = False)


fig, ax = plt.subplots(figsize=(9, 7))
years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('%Y')
ax.plot(data.index.values,data['Close/Last'],'-o',color='purple',label='Stock Close Prices')
ax.plot(data.index.values,data['Open'],'-o',color='red',label='Stock Open Prices')
ax.set(xlabel="Date",ylabel="Close/Last Prices",title="Stock Market Close/Last Price")
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
plt.legend()
plt.show()

y=data['Close/Last']
data=data.drop(['Close/Last','High','Low','Volume','Open'],axis=1)
plt.figure()
sns.distplot(y, bins=30)
plt.show()
x=data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
model=SVR(kernel='rbf', C=1e3, gamma=0.1)
#model = GridSearchCV(estimator=SVR(kernel='rbf'),param_grid={'C': [0.1, 1, 100, 1000],'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]},cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
pred=model.predict(x_test)
#new_value=scaler.fit_transform((np.array([11252019])).reshape(-1,1))
#new_prediction=model.predict(new_value)
#predictions=model.predict(scaler.fit_transform(np.array([11252019])).reshape(-1,1))
plt.figure(1, figsize=(10,5))
plt.title('Support Vector Regression | Close Stock Price vs Date')
plt.scatter(x_test, y_test, marker='o', label='Actual Stock Price')
plt.scatter(x_test, pred, color='r', label='Predicted Stock Price')
plt.xlabel('Integer Date')
plt.ylabel('Close Stock Price')
plt.legend()
plt.show()