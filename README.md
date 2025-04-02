# Inventory Forecasting System

This is a web-based inventory forecasting system that uses machine learning to predict sales based on various features.

## Features

- Interactive web interface for data input
- Real-time sales predictions
- Visual analytics with interactive charts
- Automatic model retraining with new data
- Data persistence in CSV format

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the required information in the form:
   - Item Weight
   - Item Visibility
   - Item MRP (Maximum Retail Price)
   - Outlet Establishment Year

4. Click "Predict Sales" to get the prediction

## Data

The system uses the following features for prediction:
- Item Weight
- Item Visibility
- Item MRP
- Outlet Establishment Year

The predictions are automatically saved back to the dataset for continuous learning.

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Plotly
- Bootstrap
- jQuery
