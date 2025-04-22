import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error

s3_path = 's3://my-forecasting-data-lake/dataset.csv'

data = pd.read_csv(s3_path)

regions_of_interest = ['РЕСПУБЛИКА КАЗАХСТАН']
filtered_data = data[(data['Region'].isin(regions_of_interest)) & (data['Category'] == 'Всего')]

year_columns = [col for col in filtered_data.columns if col.startswith('Year')]
value_columns = [col for col in filtered_data.columns if col.startswith('Value')]

long_data = pd.DataFrame()

for year_col, value_col in zip(year_columns, value_columns):
    temp_data = filtered_data[['Region', 'Category', year_col, value_col]].rename(
        columns={year_col: 'Year', value_col: 'Value'}
    )
    long_data = pd.concat([long_data, temp_data])

long_data.sort_values(by=['Region', 'Year'], inplace=True)
print(long_data.head(100))

scalers = {}
x_train, y_train, x_test, y_test = {}, {}, {}, {}
regions = long_data['Region'].unique()

for region in regions:
    region_data = long_data[long_data['Region'] == region]
    values = region_data['Value'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)
    scalers[region] = scaler

    sequence_length = 8
    x, y = [], []
    for i in range(sequence_length, len(values_scaled)):
        x.append(values_scaled[i-sequence_length:i, 0])
        y.append(values_scaled[i, 0])
    x, y = np.array(x), np.array(y)

    train_size = int(len(x) * 0.8)
    x_train[region], x_test[region] = x[:train_size], x[train_size:]
    y_train[region], y_test[region] = y[:train_size], y[train_size:]

models = {}

for region in regions:
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(x_train[region].shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(
        x_train[region], y_train[region],
        epochs=10, batch_size=6, validation_data=(x_test[region], y_test[region]), verbose=1
    )
    print(f"История обучения для региона {region}: {history.history}")
    models[region] = model

results = {}

for region in regions:
    y_pred = models[region].predict(x_test[region])
    y_pred_rescaled = scalers[region].inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = scalers[region].inverse_transform(y_test[region].reshape(-1, 1))

    print(f"Прогнозы для региона {region}:")
    print(y_pred_rescaled)

for region in regions:
    region_data = long_data[long_data['Region'] == region]
    plt.figure(figsize=(10, 5))
    plt.plot(region_data['Year'], region_data['Value'], label='Исходные данные')
    plt.title(f'Временные ряды для региона {region}')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()
