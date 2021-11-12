import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout



data = pd.read_csv("BTC-USD.csv", date_parser = True)
data.tail()

reqData = data[['Date','High','Low','Open','Close']]
reqData.set_index("Date",drop=True,inplace=True)

reqData['% Returns'] = reqData.Close.pct_change()
reqData['Log returns'] = np.log(1 + reqData['% Returns'])

reqData.dropna(inplace=True)

x = reqData[['Close', 'Log returns']].values

scaler = MinMaxScaler(feature_range=(0,1)).fit(x)
x_scaled = scaler.transform(x)

y = [x[0] for x in x_scaled]
split_point = int(len(x_scaled)*0.8)

x_train = x_scaled[:split_point]
x_test = x_scaled[split_point:]

y_train = y[:split_point]
y_test = y[split_point:]

time_step = 3
xtrain, ytrain, xtest, ytest = [], [], [], []

for i in range(time_step,len(x_train)):
    xtrain.append(x_train[i-time_step:i, :x_train.shape[1]])
    ytrain.append(y_train[i])

for i in range(time_step,len(y_test)):
    xtest.append(x_test[i-time_step:i, :x_test.shape[1]])
    ytest.append(y_test[i])

xtrain, ytrain = np.array(xtrain), np.array(ytrain)
xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2]))

xtest, ytest = np.array(xtest), np.array(ytest)
xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2]))

model = Sequential()

model.add(LSTM(4,input_shape=(xtrain.shape[1],xtrain.shape[2])))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")

model.fit(
    xtrain,ytrain,epochs=100,validation_data=(xtest,ytest),batch_size=16,verbose=1
)

train_predict = model.predict(xtrain)
test_predict = model.predict(xtest)

train_predict = np.c_[train_predict,np.zeros(train_predict.shape)]
test_predict = np.c_[test_predict,np.zeros(test_predict.shape)]

#train_score = mean_squared_error([x[0][0] for x in xtrain],train_predict, squared=False)
#test_score = mean_squared_error([x[0][0] for x in xtest], test_predict, squared=False)
train_predict = scaler.inverse_transform(train_predict)
train_predict = [x[0] for x in train_predict]

test_predict = scaler.inverse_transform(test_predict)
test_predict = [x[0] for x in test_predict]
print(test_predict[:5])

original_btc_price = [y[0] for y in x[split_point:]]
plt.figure(figsize=(20,10))
plt.plot(original_btc_price,color='green',label='Original BTC price')
plt.plot(test_predict,color='red',label='Predicted BTC price')
plt.title('BTC price prediction using LSTM')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()
