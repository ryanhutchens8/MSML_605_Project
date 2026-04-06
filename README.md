LSTM load forecasting w/ drift detection for Mid-Atlantic PJM zones (PEPCO, DOM, etc.)

Trains on 2018 data, runs day-by-day from 2019 and retrains whenever error goes past 3000 MW. Weather data is from Open-Meteo (DCA coords), load data from PJM DataMiner2 - data.

compile_load_data.py -> compile_weather_data.py -> manually merge into load_weather_full.csv -> simulate.py


