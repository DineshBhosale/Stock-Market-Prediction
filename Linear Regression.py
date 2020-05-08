################## LINEAR REGRESSION ##############
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import numpy as np

data=pd.read_csv('D:/NCSU/ECE 592 - Special topics in Data Science/FINAL PROJECT/googlestockdata.csv')
data=data.iloc[::-1]
data['Dates']=data.Date.str.replace("-","").astype(int)
data=data.set_index('Date')
print(data.isnull().sum())

df=data
correlation_matrix = data.corr().round(2)
print(correlation_matrix)

plt.figure(figsize=(8,8))
f, ax = plt.subplots(figsize=(8, 8))
p=data.columns
heat=sns.heatmap(data=correlation_matrix, annot=True)
heat.set_yticklabels(heat.get_yticklabels(), rotation=0)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#ax.set_yticklabels(corr_matrix.columns, rotation = 0)
#ax.set_xticklabels(corr_matrix.columns)
#sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

fig, ax = plt.subplots(figsize=(9, 7))
years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('%Y')
ax.plot(data.index.values,data['Close/Last'],'-o',color='purple',label='Stock Close Prices')
#ax.plot(data.index.values,data['Open'],'-o',color='red',label='Stock Open Prices')
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
print("Accuracy: ",model.score(x_test,y_test))
print("Coefficient(beta1): ",model.coef_)
print("Intercept(beta0): ",model.intercept_)
#date_input=input('Enter Date: MM-DD-YYY')
#date_input=np.array(int(date_input.replace("-",""))).reshape(-1,1)
#print(model.predict(date_input))
print(model.predict(np.array([11222019]).reshape(-1,1)))
pred=model.predict(x_test)
plt.figure(1, figsize=(10,5))
plt.title('Linear Regression | Close Price vs Date')
plt.scatter(x_test, y_test, marker='o', label='Actual Stock Price')
plt.plot(x_test, pred, color='r', label='Predicted Stock Price')
plt.xlabel('Integer Date')
plt.ylabel('Close Stock Price')
plt.legend()
plt.show()

