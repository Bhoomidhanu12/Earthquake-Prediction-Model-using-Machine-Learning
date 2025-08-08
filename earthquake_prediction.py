import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# 1. Load the dataset
data = pd.read_csv("data/database.csv")

# 2. Select relevant columns
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# 3. Convert Date and Time into Unix Timestamp
timestamps = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamps.append(time.mktime(ts.timetuple()))
    except ValueError:
        timestamps.append(np.nan)

data['Timestamp'] = timestamps
data.dropna(inplace=True)

# 4. Prepare final dataset
final_data = data.drop(['Date', 'Time'], axis=1)

# 5. Visualize earthquake locations on world map
def plot_earthquakes(df):
    m = Basemap(projection='mill',
                llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=-180, urcrnrlon=180,
                lat_ts=20, resolution='c')
    
    longitudes = df['Longitude'].tolist()
    latitudes = df['Latitude'].tolist()
    x, y = m(longitudes, latitudes)

    plt.figure(figsize=(12,10))
    plt.title("Earthquake Locations Worldwide")
    m.plot(x, y, 'o', markersize=2, color='blue')
    m.drawcoastlines()
    m.fillcontinents(color='coral', lake_color='aqua')
    m.drawmapboundary()
    m.drawcountries()
    plt.show()

plot_earthquakes(final_data)

# 6. Prepare training and test sets
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define neural network model builder
def create_model(neurons=16, activation='relu', optimizer='SGD', loss='squared_hinge'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# 8. Hyperparameter grid
model = KerasClassifier(build_fn=create_model, verbose=0)

param_grid = {
    'neurons': [16],
    'batch_size': [10],
    'epochs': [10],
    'activation': ['sigmoid', 'relu'],
    'optimizer': ['SGD', 'Adadelta'],
    'loss': ['squared_hinge']
}

# 9. Grid Search to find best hyperparameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print(f"Best Score: {grid_result.best_score_:.4f} with parameters: {grid_result.best_params_}")

# 10. Train final model with best params
best_params = grid_result.best_params_

final_model = Sequential()
final_model.add(Dense(best_params['neurons'], activation=best_params['activation'], input_shape=(3,)))
final_model.add(Dense(best_params['neurons'], activation=best_params['activation']))
final_model.add(Dense(2, activation='softmax'))

final_model.compile(optimizer=best_params['optimizer'], loss=best_params['loss'], metrics=['accuracy'])

final_model.fit(X_train, y_train, batch_size=best_params['batch_size'], epochs=20,
                verbose=1, validation_data=(X_test, y_test))

# 11. Evaluate the model
test_loss, test_acc = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
