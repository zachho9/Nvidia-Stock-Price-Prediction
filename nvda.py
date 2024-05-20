""" This Python script will extract Nvidia stock price (NASDAQ: NVDA) from Yahoo! Finance,
    conduct some data analysis and visualization,
    and make prediction for the next 5 trading days afterwards with Long Short-Term Memory (LSTM) model.
    This script is for learning purpose ONLY.
    NO prediction accuracy is guaranteed. DO YOUR OWN RESEARCH before investing your money.
"""

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import copy
import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class PriceChart:
    """This will create PriceChart Objects in the Web APP."""
    
    def __init__(self, df):
        """class initializer"""
        self.df = df
    
    def plot_chart(_self):
        """Plot chart with streamlit."""
        st.line_chart(_self.df)      
    

## SECTION 1: extract Nvidia stock data from Yahoo! Finance.


@st.cache_data
def get_stock_data(ticker, start_date, end_date, interval):
    """Extract stock data from Yahoo! Finance with yfinance.
       end_date is exclusive.
    """

    stock = yf.Ticker(ticker)
    stock_data = stock.history(interval=interval, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data


@st.cache_data
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


## SECTION 3: Predict Nvidia stock price


# @st.cache_data
def read_df(filename):
    """Load local csv file in case web data unobtainable."""

    df = pd.read_csv(f"{filename}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df.pop('Date')

    return df


@st.cache_data
def get_windowed_df(df, window_size=3):
    """Generate a dataframe including the date as index, the close price on the date,
       and the close prices of several days prior to the current day.
       The number of previous days included is determined by the parameter window_size.
       Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
       Accessed Date: 21-Apr-2024.
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


@st.cache_data
def split_train_val(windowed_df):
    """To simplify the project, the training set will begin at 03-Jan-2023, which was the first trading day in 2023,
       until two weeks before the last day of the existing data.
       The validation set will be from that day to the last day of the existing data.
       Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
       Accessed Date: 21-Apr-2024.
    """
    
    # training set starts from the first trading day of 2023, which is 03-Jan-2023
    train_start_row = windowed_df.index[windowed_df['Date']== '2023-01-03'].tolist()[0]

    # training set ends at two weeks before the last day of the existing data.
    train_end_row = windowed_df.iloc[-14].name

    # Slice and reshape data from the dataframe for model fitting
    windowed_df_np = windowed_df.to_numpy()[train_start_row:]

    windowed_date_np = windowed_df_np[:, 0]
    windowed_X_np = windowed_df_np[:, 1:-1].reshape(len(windowed_df_np[:, 1:-1]), windowed_df_np[:, 1:-1].shape[1], 1).astype(np.float32)
    windowed_y_np = windowed_df_np[:, -1].astype(np.float32)

    # Split train and validation set
    len_train = train_end_row - train_start_row

    dates_train, X_train, y_train = windowed_date_np[:len_train], windowed_X_np[:len_train], windowed_y_np[:len_train]
    dates_val, X_val, y_val = windowed_date_np[len_train:], windowed_X_np[len_train:], windowed_y_np[len_train:]

    print(f"Train set shape:       dates_train {dates_train.shape},  X_train {X_train.shape},  y_train {y_train.shape}")
    print(f"Validation set shape:  dates_val {dates_val.shape},      X_val {X_val.shape},      y_val {y_val.shape}")

    return dates_train, X_train, y_train, dates_val, X_val, y_val


@st.cache_resource
def build_model(X_train):
    """Stack LSTM Model.
       Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
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

    # model.summary()

    # Save model
    model.save("lstm_model_nvda.keras")

    return model


@st.cache_resource
def pred_train(_model, X_train, y_train, X_val, y_val):
    """Fit and predict training set.
       Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
       Accessed Date: 21-Apr-2024.
    """
    
    _model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    train_predictions = _model.predict(X_train).flatten()
    val_predictions = _model.predict(X_val).flatten()

    return train_predictions, val_predictions


@st.cache_resource
def get_windowed_df_unseen(_model, df, num_new_days=5, window_size=3):
    """Generate predicted data for next 5 trading days.
       Reference: https://www.youtube.com/watch?v=CbTU92pbDKw
       Accessed Date: 21-Apr-2024.
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
        X_last_day_price = price_for_predict.reshape(1, price_for_predict.shape[0], 1).astype(np.float32)
        y_new_day_price = _model.predict(X_last_day_price).flatten()

        # Merge new price data with df
        # New row starts with Date
        unseen_row_df = {'Date': [next_biz_day]}

        # Enumerate X_last_day_price array for each T-j column
        for j, price in enumerate(X_last_day_price.flatten()):
            unseen_row_df[f'T-{window_size-j}'] = [price]

        # Close price for new row
        unseen_row_df['Close'] = y_new_day_price

        # Concatenate dataframe
        windowed_df_unseen = pd.concat([df, pd.DataFrame(unseen_row_df)], ignore_index=True)

        # Update the last day row value and df
        last_day = next_biz_day
        price_for_predict = windowed_df_unseen.to_numpy()[-1, 2:]
        df = windowed_df_unseen

    return windowed_df_unseen


# Streamlit Session State Function 1, callback in st.button
def click_button_switch():
    """Reference: https://docs.streamlit.io/develop/concepts/design/buttons
       Date: 17-May-2024
    """
    st.session_state.button = not st.session_state.button


# Streamlit Session State Function 2, callback in st.date_input
def date_changed():
    """Reference: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
       Date: 17-May-2024
    """
    if st.session_state.date != st.session_state.prev_date:
        st.session_state.button = False
        st.session_state.prev_date = st.session_state.date


def run_app():
    """Main function to run the program"""

    st.title("Nvidia Stock Price Prediction :chart_with_upwards_trend:")
    st.write("Please note:")
    st.write("- Date format is YYYY-MM-DD.\n - The data starts from 2000-01-01.\n - Please select a valid End Date to proceed.\n - The data contained trading day only.")
    
    
    ### SECTION 1 - Extract data and save to csv
    
    
    st.header('Section 1 - Extract Stock Price Data', divider='rainbow')
    st.subheader("Please choose an End Date:")

    # Initialize session state for streamlit
    if 'prev_date' not in st.session_state:
        st.session_state.prev_date = ""
    if 'button' not in st.session_state:
        st.session_state.button = False
    
    today = datetime.date.today()
    end_date = st.date_input("Choose an End Date after 2000-01-01, but before Today! Error will occur if you don't. Try if you insist:", value='today', key='date', on_change=date_changed, format='YYYY-MM-DD')
    
    if end_date > datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date() and end_date <= today:
        st.write("The End Date you select is:", end_date, ". But End Date is not included in the full data.")
        st.markdown('***You can change the End Date multiple times to see different datasets and predictions, have fun!***')
    elif end_date > today:
        st.write("Bummer! We are not a time machine. Can't fetch future data! Please select a valid date.")
        return
    else:
        st.write("Cool! You successfully select a date before 2000-01-01. However, your request cannot be proceeded. Please select a valid date.")
        return
    
    st.button('Click to Run or Stop', on_click=click_button_switch, type='primary')
    
    if st.session_state.button:
        
        st.write('App is Running!')
        
        stock_data = get_stock_data('NVDA', '2000-01-01', end_date, '1d')
        df, filename = convert_date_column(stock_data, 'nvda_close')
       
        # Read data and from saved csv.
        filename = 'nvda_close'
        df = read_df(filename)
        
        # Plot the whole price chart
        st.subheader("Have a look at the chart, and play with it:")
        st.write(f"Nvidia Stock Price Chart from 2000-01-01 to {df.index[-1].date()}:")
        full_chart = PriceChart(df)
        full_chart.plot_chart()
        
    
        ###  SECTION 2 - Exploratory Stock Analysis
        
        
        st.header('Section 2 - Stock Trend Analysis', divider='rainbow')

        # Reload file
        filename = 'nvda_close'
        df = read_df(filename)
        
        # Use streamlit slider to explore any time range within the full data.
        st.subheader('Move the slider to choose a time range:')
        start_date, end_date = st.slider('', 
                                        min_value=df.index[0].date(),
                                        max_value=df.index[-1].date(),
                                        value=(df.index[0].date(), df.index[-1].date()),
                                        format='YYYY-MM-DD')
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_se2 = copy.deepcopy(df)
        df_se2 = df_se2.loc[start_date:end_date]
        st.write('Nvidia Stock Price Chart in the selected range:')
        se2_chart = PriceChart(df_se2)
        se2_chart.plot_chart()
        
        # Showcase the key indicators for stock analysis
        st.subheader('Have a look at these Key Indicators:')
        
        start_price_in_range = df_se2.iloc[0,0]
        end_price_in_range = df_se2.iloc[-1,0]
        max_price_in_range = df_se2.iloc[:,0].max()
        min_price_in_range = df_se2.iloc[:,0].min()
        
        start_price_date = df_se2.index[df_se2['Close'] == start_price_in_range][0].date()
        end_price_date = df_se2.index[df_se2['Close'] == end_price_in_range][0].date()
        max_price_date = df_se2.index[df_se2['Close'] == max_price_in_range][0].date()
        min_price_date = df_se2.index[df_se2['Close'] == min_price_in_range][0].date()
        
        df_se2_range = pd.DataFrame({'Indicators': ['Start Price', 'End Price', 'Max Price', 'Min Price'],
                                    'Prices': [start_price_in_range, end_price_in_range, max_price_in_range, min_price_in_range],
                                    'On Dates': [start_price_date, end_price_date, max_price_date, min_price_date]})
        df_se2_range['Prices'] = df_se2_range['Prices'].round(4).apply(lambda x: f"{x:.4f}")
        st.dataframe(df_se2_range, use_container_width=True)
        
        st.subheader('The ROI (Return on Investment) in the selected range:')
        
        # Holding Price Return
        holding_period_return = (end_price_in_range - start_price_in_range) / start_price_in_range * 100
        st.write(f"- Holding Period Return (buy at start date, sell at end date): {holding_period_return:.2f}%")
        
        # Maximum Gain or Maximum Loss
        if max_price_date < min_price_date:
            max_gain_loss = (min_price_in_range - max_price_in_range) / max_price_in_range * 100
        else:
            max_gain_loss = (max_price_in_range - min_price_in_range) / min_price_in_range * 100

        st.write(f"- Maximum Gain / Maximum Loss (difference between max and min price): {max_gain_loss:.2f}%")
        
        
        ###  SECTION 3 - Stock Price Prediction with LSTM
    
    
        st.header('Section 3 - Stock Price Prediction with LSTM', divider='rainbow')
    
        st.markdown('***Stock prediction is not easy, it will take a while...***')

        # Reload file
        filename = 'nvda_close'
        df = read_df(filename)
        
        # Prepare data for modeling
        windowed_df = get_windowed_df(df, window_size=3)
        dates_train, X_train, y_train, dates_val, X_val, y_val = split_train_val(windowed_df)
        
        # LSTM model building, training, and predicting
        model = build_model(X_train)
        train_predictions, val_predictions = pred_train(model, X_train, y_train, X_val, y_val)

        # Plot actual and predicted result for training and validation set
        df_train_actual = pd.concat([pd.DataFrame({'Dates_Training': dates_train}), pd.DataFrame({'Training_Actual': y_train})], axis=1)
        df_val_actual = pd.concat([pd.DataFrame({'Dates_Validation': dates_val}), pd.DataFrame({'Validation_Actual': y_val})], axis=1)
        df_train_pred = pd.concat([pd.DataFrame({'Dates_Training': dates_train}), pd.DataFrame({'Training_Prediction': train_predictions})], axis=1)
        df_val_pred = pd.concat([pd.DataFrame({'Dates_Validation': dates_val}), pd.DataFrame({'Validation_Prediction': val_predictions})], axis=1)

        # Learn from https://discuss.streamlit.io/t/plot-multiple-line-chart-in-a-single-line-chart/66339/9
        # And https://python-graph-gallery.com/522-plotly-customize-title/
        # And https://plotly.com/python/creating-and-updating-figures/
        # Date: 13-May-2024
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train_actual['Dates_Training'], y=df_train_actual['Training_Actual'], mode='lines+markers', name='Training_Actual'))
        fig.add_trace(go.Scatter(x=df_val_actual['Dates_Validation'], y=df_val_actual['Validation_Actual'], mode='lines+markers', name='Validation_Actual'))
        fig.add_trace(go.Scatter(x=df_train_pred['Dates_Training'], y=df_train_pred['Training_Prediction'], mode='lines+markers', name='Training_Prediction'))
        fig.add_trace(go.Scatter(x=df_val_pred['Dates_Validation'], y=df_val_pred['Validation_Prediction'], mode='lines+markers', name='Validation_Prediction'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Stock Price (USD)', legend_title='Legend')
        
        st.subheader('Have a look at the model performance, interactively:')
        st.plotly_chart(fig)
        
        # Print predicted data on unseen data
        st.subheader('Now let\'s predict for the next 5 trading days:')
        st.write('The accuracy will reduce significantly if predicting too long afterwards.')
        
        windowed_df_unseen = copy.deepcopy(windowed_df)
        new_df_unseen = get_windowed_df_unseen(model, df=windowed_df_unseen, num_new_days=5, window_size=3)
        st.dataframe(new_df_unseen[['Date', 'Close']].tail(5), use_container_width=True)

        st.markdown('**Reminder! The prediction is just for fun. Do Your Own Research before investing your money.**')

    else:
        st.write('App not Running...')
    
run_app()
