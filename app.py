import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Prediction')
user_input = st.text_input('Enter Stock Name','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#data
st.subheader('Data from 2010-2019')
st.write(df.describe())

#graphs
st.subheader('Closing Price vs Time (with 100 moving average)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time (with 100 and 200 moving average)')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training =  pd.DataFrame(df["Close"][0:int(len(df)*(0.7))])
data_testing =  pd.DataFrame(df["Close"][int(len(df)*(0.7)):int(len(df))])

from sklearn.preprocessig import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 

data_training_array = scaler.fit_transform(data_training)

# x_train = []
# y_train = []

# for i in range(100,data_training.shape[0]):
#     x_train.apend(data_training_array[i-100: i])
#     x_train.apend(data_training_array[i,0])

# x_train, ytrain = np.array(x_train), np.array(y_train) 

model = load_model("spm.h5")

#Testing 
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.apend(input_data[i-100: i])
    x_test.apend(input_data[i,0])

x_test, ytest = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Visualisation
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = "Original Price")
plt.plot(y_predicted,'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
