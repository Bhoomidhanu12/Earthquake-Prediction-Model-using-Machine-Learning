# Earthquake Prediction Model Using Machine Learning

This project implements an earthquake prediction model using a neural network built with Keras and TensorFlow. It utilizes historical earthquake data to predict earthquake magnitude and depth based on timestamp and location (latitude and longitude).

---

## Project Overview

Predicting earthquakes precisely remains a challenge in earth sciences. This project leverages machine learning techniques on seismic data to forecast earthquake characteristics, helping to identify potential high-risk regions.

---

## Dataset

The dataset contains earthquake event records with features such as:

- Date and Time of the event
- Latitude and Longitude (location)
- Depth of the earthquake
- Magnitude of the earthquake

The dataset file (`database.csv`) should be placed inside the `data/` directory.

---

## Features

- Data preprocessing including conversion of date and time to Unix timestamp
- Visualization of earthquake locations on a world map using Basemap
- Train/test data split for model validation
- Neural network model with hyperparameter tuning using GridSearchCV
- Model evaluation with loss and accuracy metrics

---

## Installation
1. Locally on your computer
Use Python installed on your PC with an IDE or editor like:
VS Code
 Clone the repository:
   ```bash
   git clone https://github.com/your-username/earthquake-prediction-ml.git
   cd earthquake-prediction-ml
Install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Run the main script to train and evaluate the model:

bash
Copy
Edit
python earthquake_prediction.py
The script will first display a map plotting earthquake occurrences.

Then, it will perform hyperparameter tuning to find the best model configuration.

Finally, it will train the final model and output the test loss and accuracy.

Requirements
Python 3.7+

numpy

pandas

matplotlib

tensorflow

keras

scikit-learn

basemap

basemap-data-hires

Note: Installing basemap might require additional system dependencies depending on your OS.

Project Structure
graphql
Copy
Edit
earthquake-prediction-ml/
 ├── data/
 │    └── database.csv      # Earthquake data CSV file
 ├── earthquake_prediction.py  # Main Python script
 ├── requirements.txt       # Python dependencies
 └── README.md              # Project documentation
Troubleshooting
If you face issues with Basemap installation, consider using cartopy as an alternative for visualization.

Ensure you run the script with the same Python environment where dependencies are installed.

For issues related to TensorFlow/Keras wrappers, verify imports and package versions.
In a Jupyter Notebook
Jupyter notebooks are ideal because you can run code cells one by one and see outputs (like plots) inline.
You can install Jupyter by running:
pip install notebook
Run notebook with:
jupyter notebook
For Working project in **Jupyter Refer The Documentation provided**
###License
This project is open-source and available under the MIT License.
