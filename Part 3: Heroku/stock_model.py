import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, send_file, request
import base64
from io import BytesIO
import os


app = Flask(__name__)

@app.route('/')
def form():

    """Load the pre-saved model and data to test"""

    model = load_model('BTC-predict.h5')
    stock = yf.Ticker("BTC-USD")


    """Create training and test dataset. Training dataset is
    80% of the total data and the remaining 20% will be predicted,
    The model is already trained. We just need this 80% finishing line to
    sst the prediction starting point"""

    hist = stock.history(period="5y")
    hist.tail(10)
    df=hist
    d=30
    n=int(hist.shape[0]*0.8)
    sc = MinMaxScaler(feature_range = (0, 1))

    dataset_train = df.iloc[:n, 1:2]
    dataset_train_scaled = sc.fit_transform(dataset_train)
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - d:].values


    """Scale and reshape the data"""

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
