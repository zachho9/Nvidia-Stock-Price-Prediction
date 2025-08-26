# Note:
The Streamlit application is currently not working (likely API error, as of August 2025). Please refer to the video demonstration of the application here: https://youtu.be/xmetGvT99fw

# Nvidia-Stock-Price-Prediction
## Toolkit
- Language: **Python**
- Libraries:
  - Extract Nvidia Stock Data: **yfinance**
  - Data Wrangling and Visualization: **Pandas**, **NumPy**, **Plotly**
  - LSTM Modeling and Prediction: **TensorFlow Keras**
  - Dashboard App Dev: **Streamlit**
## Streamlit Web Dashboard:
- Link: https://nvidia-stock-price-prediction.streamlit.app/
- If it falls asleep, simply click the button on that screen to wake it up.
## Run on Local Machine: 
- `conda create --name <ENV_NAME> --file requirements.txt`
- `conda activate <ENV_NAME>`
- `python -m streamlit run nvda.py`
