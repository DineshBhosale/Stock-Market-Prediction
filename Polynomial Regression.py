################## LINEAR REGRESSION ############
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

data=pd.read_csv('D:/NCSU/ECE 592 - Special topics in Data Science/FINAL PROJECT/googlestockdata.csv')
data=data.iloc[::-1]
data['Dates']=data.Date.str.replace("-","").astype(int)
data=data.set_index('Date')
print(data.isnull().sum())

correlation_matrix = data.corr().round(2)
print(correlation_matrix)
plt.figure(2,figsize=(7,7))
sns.heatmap(data=correlation_matrix, annot=True, yticklabels=False)

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
model=LinearRegression()
poly=PolynomialFeatures(degree=6)
xtrain_poly=poly.fit_transform(x_train)
poly.fit(xtrain_poly,y_train)
model.fit(xtrain_poly,y_train)
pred=model.predict(poly.fit_transform(x_test))
#date_input=input('Enter Date: MM-DD-YYY')
#date_input=np.array(int(date_input.replace("-",""))).reshape(-1,1)
#print("Prediction",model.predict(poly.fit_transform(date_input)))
print(model.predict(poly.fit_transform(np.array([11292019]).reshape(-1,1))))
pred=pd.DataFrame(pred,index=x_test.index)
print('Accuracy: ',r2_score(y_test,pred))
plt.figure(1, figsize=(9,7))
plt.title('Polynomial Regression | Close Price vs Date')
plt.scatter(x_test, y_test, marker='o', label='Actual Close Price')
plt.scatter(x_test, pred,color='red', label='Predicted Close Price')
plt.xlabel('Integer Date')
plt.ylabel('Close Stock Price')
plt.legend()
plt.show()



