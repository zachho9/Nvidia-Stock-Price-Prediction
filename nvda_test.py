"""Predict Nvidia stock price within one week after the last day of the extracted data.
   This script is a demonstration of Python skill. NO prediction accuracy is guaranteed.
"""
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import copy
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""Section 1: extract Nvidia stock data from Yahoo! Finance."""


def get_stock_data(ticker, start_date, end_date, interval):
    """Extract stock data from Yahoo Finance with yfinance. end_date is not inclusive."""

    stock = yf.Ticker(ticker)
    stock_data = stock.history(
        interval=interval, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data


def convert_date_column(stock_data, filename):
    """Convert Data Column datatype."""

    # Convert to datetime64 but without timestamp using pandas methods
    df = stock_data.copy()
    df['Date'] = df['Date'].dt.date
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract Date and Close
    df = df[['Date', 'Close']]

    # Replace index with Date
    df.index = df.pop('Date')

    # Write to csv file to save progress
    df.to_csv(f"{filename}.csv")

    return df, filename


"""Section 2: Conduct Exploratory Data Analysis"""


"""Section 3: Predict Nvidia stock price"""


def read_df(filename):
    """Load local csv file in case web data unobtainable."""

    df = pd.read_csv(f"{filename}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df.pop('Date')

    return df

# Define a function to generate a windowed dataframe for LSTM prediction


def get_windowed_df(df, window_size=3):
    """Generate a dataframe including the date as index, the close price on the date,
       and the close prices of several days prior to the current day.
       The number of previous days included is determined by the parameter window_size.
    """
    X = []
    y = []
    date_id = []
    windowed_df = pd.DataFrame()

    for i in range(window_size, len(df)):
        date_id.append(df.index[i])
        X.append(df.iloc[i-window_size:i, 0])
        y.append(df.iloc[i, 0])

    windowed_date = pd.DataFrame(np.array(date_id))
    windowed_X = pd.DataFrame(np.array(X))
    windowed_y = pd.DataFrame(np.array(y))

    windowed_df = pd.concat([windowed_date, windowed_X, windowed_y], axis=1)
    X_col_names = [f"T-{window_size-j}" for j in range(window_size)]
    windowed_df.columns = ['Date'] + X_col_names + ['Close']

    return windowed_df


def split_train_val(windowed_df):
    """To simplify the project, the training set will begin at 03-Jan-2023, which was the first trading day in 2023,
       until two weeks before the last day of the existing data.
       The validation set will be from that day to the last day of the existing data.
       Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
       Accessed Date: 21-Apr-2024.
    """
    # training set starts from the first trading day of 2023, which is 03-Jan-2023
    train_start_row = windowed_df.index[windowed_df['Date']
                                        == '2023-01-03'].tolist()[0]

    # training set ends at two weeks before the last day of the existing data.
    train_end_row = windowed_df.iloc[-14].name

    # Slice and reshape data from the dataframe for model fitting
    windowed_df_np = windowed_df.to_numpy()[train_start_row:]

    windowed_date_np = windowed_df_np[:, 0]
    windowed_X_np = windowed_df_np[:, 1:-1].reshape(len(
        windowed_df_np[:, 1:-1]), windowed_df_np[:, 1:-1].shape[1], 1).astype(np.float32)
    windowed_y_np = windowed_df_np[:, -1].astype(np.float32)

    # Split train and validation set
    len_train = train_end_row - train_start_row

    dates_train, X_train, y_train = windowed_date_np[:
                                                     len_train], windowed_X_np[:len_train], windowed_y_np[:len_train]
    dates_val, X_val, y_val = windowed_date_np[len_train:
                                               ], windowed_X_np[len_train:], windowed_y_np[len_train:]

    print(
        f"Train set shape:       dates_train {dates_train.shape},  X_train {X_train.shape},  y_train {y_train.shape}")
    print(
        f"Validation set shape:  dates_val {dates_val.shape},      X_val {X_val.shape},      y_val {y_val.shape}")

    return dates_train, X_train, y_train, dates_val, X_val, y_val


def build_model(X_train):
    """Model Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
       Accessed Date: 21-Apr-2024.
    """
    # Build LSTM model
    model = Sequential([layers.Input((X_train.shape[1], 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(16, activation='relu'),
                        layers.Dense(16, activation='relu'),
                        layers.Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])

    model.summary()

    # Save model
    model.save("lstm_nvda.keras")
    print("Model Created and Saved")
    return model


def pred_train(model, dates_train, X_train, y_train, dates_val, X_val, y_val):
    """Fit and predict training set"""
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    train_predictions = model.predict(X_train).flatten()

    val_predictions = model.predict(X_val).flatten()

    return train_predictions, val_predictions


# Define a function to generate predicted data for next 5 business days
def get_windowed_df_unseen(model, df, num_new_days=5, window_size=3):
    """Adding number of days(rows) to the df,
       then predict the close price with LSTM model
       to generate future data.
    """

    # Get the last day and its price of known data
    last_day = df.iloc[-1, 0]
    price_for_predict = df.to_numpy()[-1, 2:]

    # Add number of rows, skipping Sat(weekday 5) and Sun(weekday 6)
    for i in range(num_new_days):
        next_biz_day = last_day + pd.DateOffset(days=1)

        # When Sat or Sun, +1 day
        while next_biz_day.weekday() >= 5:
            next_biz_day += pd.DateOffset(days=1)

        # Predict close price on the new date(row))
        X_last_day_price = price_for_predict.reshape(
            1, price_for_predict.shape[0], 1).astype(np.float32)
        y_new_day_price = model.predict(X_last_day_price).flatten()

        # Merge new price data with df
        # New row starts with Date
        unseen_row_df = {'Date': [next_biz_day]}

        # Enumerate X_last_day_price array for each T-j column
        for j, price in enumerate(X_last_day_price.flatten()):
            unseen_row_df[f'T-{window_size-j}'] = [price]

        # Close price for new row
        unseen_row_df['Close'] = y_new_day_price

        # Concatenate dataframe
        windowed_df_unseen = pd.concat(
            [df, pd.DataFrame(unseen_row_df)], ignore_index=True)

        # Update the last day row value and df
        last_day = next_biz_day
        price_for_predict = windowed_df_unseen.to_numpy()[-1, 2:]
        df = windowed_df_unseen

    return windowed_df_unseen


"""Below for code test only"""


def main():
    """for code test only"""

    # stock_data = get_stock_data('NVDA', '2000-01-01', '2024-04-06', '1d')
    # df, filename = convert_date_column(stock_data, 'nvda_close_only_test')
    filename = 'nvda_close_only_test'
    df = read_df(filename)
    windowed_df = get_windowed_df(df, window_size=3)
    dates_train, X_train, y_train, dates_val, X_val, y_val = split_train_val(
        windowed_df)
    model = build_model(X_train)
    train_predictions, val_predictions = pred_train(
        model, dates_train, X_train, y_train, dates_val, X_val, y_val)
    windowed_df_unseen = copy.deepcopy(windowed_df)
    new_df_unseen = get_windowed_df_unseen(
        model, df=windowed_df_unseen, num_new_days=5, window_size=3)

    # streamlit plot training and validation result
    df_train_real = pd.concat(
        [pd.DataFrame({'dates_train': dates_train}), pd.DataFrame({'y_train': y_train})], axis=1)
    df_val_real = pd.concat(
        [pd.DataFrame({'dates_val': dates_val}), pd.DataFrame({'y_val': y_val})], axis=1)
    df_train_pred = pd.concat(
        [pd.DataFrame({'dates_train': dates_train}), pd.DataFrame({'train_predictions': train_predictions})], axis=1)
    df_val_pred = pd.concat(
        [pd.DataFrame({'dates_val': dates_val}), pd.DataFrame({'val_predictions': val_predictions})], axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_train_real['dates_train'],    y=df_train_real['y_train'],             mode='lines',   name='y_train'))
    fig.add_trace(go.Scatter(
        x=df_val_real['dates_val'],        y=df_val_real['y_val'],                 mode='lines',   name='y_val'))
    fig.add_trace(go.Scatter(
        x=df_train_pred['dates_train'],    y=df_train_pred['train_predictions'],   mode='lines',   name='train_predictions'))
    fig.add_trace(go.Scatter(
        x=df_val_pred['dates_val'],        y=df_val_pred['val_predictions'],       mode='lines',   name='val_predictions'))

    fig.update_layout(xaxis_title='Date', yaxis_title='Value',
                      legend_title='Legend')
    st.plotly_chart(fig)
    # streamlit write predicted data
    st.write(new_df_unseen.tail(10))


main()
