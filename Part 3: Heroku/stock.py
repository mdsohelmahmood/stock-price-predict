from flask import Flask, render_template, send_file, request
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from array import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import yfinance as yf



app = Flask(__name__)

"""The default page will route to the form.html page where user can input
necessary variables for machine learning"""

@app.route('/')
def form():
    return render_template('form.html')

""" Once the data is keyed in by the user and the submit button is pressed,
the user will have to wait for the training of the model depending on the
epoch number. Once trained, the model will show the predicted output of this
time series data."""


@app.route('/data', methods=['POST'])
def hello():

    """Get the input names from form.html"""

    stock_name = request.form['Name']
    ep = request.form['Epochs']
    ahead = request.form['Ahead']
    d = request.form['Days']

    ep = int(ep)
    ahead = int(ahead)
    d = int(d)
    stock = yf.Ticker(stock_name)


    """Parse historical data of 5yrs from Yahoo Finance"""

    hist = stock.history(period="5y")


    """Create training and test dataset. Training dataset is
    80% of the total data and the remaining 20% will be predicted"""

    df=hist
    n=int(hist.shape[0]*0.8)
    training_set = df.iloc[:n, 1:2].values
    test_set = df.iloc[n:, 1:2].values


    """Scale and reshape the data"""

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(d, n-ahead):
        X_train.append(training_set_scaled[i-d:i, 0])
        y_train.append(training_set_scaled[i+ahead, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    """Creae the model will neural network having 3 layers of LSTM.
    Add the LSTM layers and some Dropout regularisation"""

    model = Sequential()

    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))


    """Compile and fit the model"""

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(X_train, y_train, epochs = ep, batch_size = 32)


    """Once the model is created, it can be saved. Proceeding forward,
    we need to find out the starting point based on the user inputs for
    "Ahead" and "Days". The data is reshaped next."""

    dataset_train = df.iloc[:n, 1:2]
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - d:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(d, inputs.shape[0]):
        X_test.append(inputs[i-d:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    """Predict with the model on the test data"""

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    df['Date']=df.index
    df=df.reset_index(drop=True)


    """Plot the actual and predicted data"""

    plt.plot(df.loc[n:, 'Date'],dataset_test.values, color = 'red', label = 'Actual Price')
    plt.plot(df.loc[n:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted Price')
    # plt.xticks(np.arange(0,459,50))
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=90)
    STOCK = BytesIO()
    plt.savefig(STOCK, format="png")



    """Send the plot to plot.html"""

    STOCK.seek(0)
    plot_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
    return render_template("plot.html", plot_url=plot_url)
