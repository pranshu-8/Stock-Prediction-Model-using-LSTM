import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data

start = '2010-01-01'
end = '2019-12-31'
df = data.DataReader('AAPL','yahoo',start,end)
df = df.reset_index()
df = df.drop(['Date','Adj Close'],axis = 1)
plt.plot(df.Close)
ma100 = df.Close.rolling(100).mean()



data_training =  pd.DataFrame(df["Close"][0:int(len(df)*(0.7))])
data_testing =  pd.DataFrame(df["Close"][int(len(df)*(0.7)):int(len(df))])
from sklearn.preprocessig import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 
data_training_array = scaler.fit_transform(data_training)
x_train = []
y_train = []

for i in range(100,data_training.shape[0]):
    x_train.apend(data_training_array[i-100: i])
    x_train.apend(data_training_array[i,0])

x_train, ytrain = np.array(x_train), np.array(y_train)     

#ML Model :

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model= Sequential()

#First Input layer and LSTM layer with 0.2% dropout
model.add(LSTM(units=50,return_sequences=True,activation = 'relu',input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

# Where:
#     return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.

# Second LSTM layer with 0.2% dropout
model.add(LSTM(units=60,activation = 'relu',return_sequences=True))
model.add(Dropout(0.3))

#Third LSTM layer with 0.2% dropout
model.add(LSTM(units=80,activation = 'relu',return_sequences=True))
model.add(Dropout(0.4))

#Fourth LSTM layer with 0.2% dropout, we wont use return sequence true in last layers as we dont want to previous output
model.add(LSTM(units=120,activation = 'relu'))
model.add(Dropout(0.5))
#Output layer , we wont pass any activation as its continous value model
model.add(Dense(units=1))

#Compiling the network
model.compile(optimizer='adam',loss='mean_squared_error')


#fitting the network
model.fit(x_train,y_train,epochs=50)
model.save('spm.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.apend(input_data[i-100: i])
    x_test.apend(input_data[i,0])

x_test, ytest = np.array(x_test), np.array(y_test)

#predictions
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = "Original Price")
plt.plot(y_predicted,'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel("Price")
plt.legend()
plt.show()