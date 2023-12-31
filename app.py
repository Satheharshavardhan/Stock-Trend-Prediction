import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import load_model
from keras.models import load_model
import streamlit as st

start_date = datetime(2010, 1, 1)
end_date = datetime(2023, 12, 30)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter the name of Stock', 'AAPL')
stock_data = yf.download(user_input, start=start_date, end=end_date)

# Description
st.subheader('Data from 1st Jan, 2010 to 30th Dec,2023')
st.write(stock_data.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(10, 6))
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 days MA')
fig = plt.figure(figsize=(10, 6))
ma100 = stock_data.Close.rolling(100).mean()
plt.plot(ma100, c='r')
plt.plot(stock_data.Close)
plt.legend(['Closing Price', 'ma100'])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 and 200 days MA')
fig = plt.figure(figsize=(10, 6))
ma200 = stock_data.Close.rolling(200).mean()
plt.plot(ma100, c='y')
plt.plot(ma100, c='r')
plt.plot(stock_data.Close)
plt.legend(['Closing Price', 'ma100', 'ma200'])
st.pyplot(fig)

# splitting the data into training and testing
trained_data = pd.DataFrame(stock_data['Close'][:int(len(stock_data)*0.70)])
test_data = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.70):])

# Normalizing the data
scalar = MinMaxScaler(feature_range=(0, 1))
transformed_train_data = scalar.fit_transform(trained_data)

# Loading the model
model = load_model('lstm.h5')

# Testing the model

past_100_days = trained_data.tail(100)
final = past_100_days.append(test_data, ignore_index=True)
input_data = scalar.fit_transform(final)
X_test = []
Y_test = []
for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    Y_test.append(input_data[i, 0])
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# predicting the model
Y_predict = model.predict(X_test)
factor = scalar.scale_
factor = 1/factor[0]
Y_predict = Y_predict * factor
Y_test = Y_test * factor

# Plottinf the final graph
st.subheader('Predictions vs actual values')
fig2 = plt.figure(figsize=(10, 6))
plt.plot(Y_test, c='r', label='Original Price')
plt.plot(Y_predict, c='g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(fig=fig2)
