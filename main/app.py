from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json
import plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta
import locale
import os
import csv

# Set locale for currency format
try:
    locale.setlocale(locale.LC_ALL, 'en_IN')
except locale.Error:
    try:
        # Try English locale as fallback
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            # Try system default locale
            locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            print("Warning: Could not set locale. Using default currency formatting.")

def format_currency(amount):
    """Format amount as currency with fallback options"""
    try:
        return locale.currency(amount, grouping=True)
    except:
        # Fallback to simple formatting
        return f"₹{amount:,.2f}"

app = Flask(__name__)

# Constants
USD_TO_INR = 82.3
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June']
PREDICTION_HISTORY_FILE = 'prediction_history.csv'

# Initialize global variables
print("Loading data...")
df = pd.read_csv('inventory_data.csv')

# Fix timestamp parsing
try:
    print("Converting timestamps...")
    # Try parsing with microseconds first
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    # For any failed rows, try without microseconds
    mask = df['Timestamp'].isna()
    if mask.any():
        print(f"Retrying {mask.sum()} timestamps without microseconds...")
        df.loc[mask, 'Timestamp'] = pd.to_datetime(
            df.loc[mask, 'Timestamp'], 
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
    
    # Drop any remaining invalid timestamps
    invalid_timestamps = df['Timestamp'].isna()
    if invalid_timestamps.any():
        print(f"Found {invalid_timestamps.sum()} invalid timestamps. Removing these rows.")
        df = df.dropna(subset=['Timestamp'])
    
    print("Timestamp conversion successful")
    print(f"First timestamp: {df['Timestamp'].min()}")
    print(f"Last timestamp: {df['Timestamp'].max()}")
except Exception as e:
    print(f"Error converting timestamps: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

# Load and preprocess data
def load_data():
    global df
    df['Month'] = pd.Categorical(df['Month'], categories=MONTHS, ordered=True)
    df['Item_MRP_INR'] = df['Item_MRP'] * USD_TO_INR
    return df

load_data()

model = None
feature_columns = None

def train_model():
    """Train model with enhanced features and hyperparameter tuning"""
    global df, model, feature_columns
    
    try:
        print("Starting model training...")
        
        # Prepare features
        numerical_features = ['Item_Weight', 'Item_MRP_INR', 'Shelf_Life_Days', 'Trending_Score', 
                            'Establishment_Year', 'Visibility']
        
        # Create feature matrix
        X = df[numerical_features].copy()
        
        # Add month encoding
        month_dummies = pd.get_dummies(df['Month'], prefix='Month')
        X = pd.concat([X, month_dummies], axis=1)
        
        # Create dummy variables for categorical features
        categorical_features = {
            'Category': ['Dairy Products', 'Fruits and Vegetables', 'Grains and Cereals', 'Packaged Snacks'],
            'Seasonal_Demand': ['High', 'Low', 'Medium'],
            'Packaging_Type': ['Loose', 'Packaged'],
            'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
            'Fat_Content': ['Full Fat', 'Low Fat', 'Regular']
        }
        
        for feature, values in categorical_features.items():
            dummies = pd.get_dummies(df[feature], prefix=feature)
            X = pd.concat([X, dummies], axis=1)
        
        # Target variable
        y = df['Item_Outlet_Sales']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X, y)
        feature_columns = X.columns.tolist()
        
        print("Model training completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return False

# Load or train model
try:
    print("Attempting to load model and features...")
    model = joblib.load('inventory_model.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    print("Model and features loaded successfully")
    print("Feature columns:", feature_columns)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Training new model...")
    train_model()
    joblib.dump(model, 'inventory_model.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    print("New model saved")

def store_prediction(data):
    """Store prediction data in CSV file"""
    file_exists = os.path.isfile(PREDICTION_HISTORY_FILE)
    
    try:
        with open(PREDICTION_HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'category', 'item_name', 
                                                 'timeline', 'demand', 'mrp_inr'])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': data['timestamp'],
                'category': data['category'],
                'item_name': data['item_name'],
                'timeline': data['timeline'],
                'demand': data['demand'],
                'mrp_inr': data['mrp_inr']
            })
            
        return True
    except Exception as e:
        print(f"Error storing prediction: {str(e)}")
        return False

@app.route('/store_prediction', methods=['POST'])
def handle_store_prediction():
    """Handle storing prediction data"""
    try:
        data = request.get_json()
        success = store_prediction(data)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to store prediction'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Retrain model with updated data"""
    try:
        # Load prediction history
        if os.path.isfile(PREDICTION_HISTORY_FILE):
            history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
            
            # Combine with existing data
            combined_df = pd.concat([df, history_df], ignore_index=True)
            
            # Train model with combined data
            global model
            model = train_model(combined_df)
            
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_model_status')
def get_model_status():
    """Get current model status and metrics"""
    try:
        r2_score, rmse = get_model_metrics()
        
        # Get prediction history stats
        history_count = 0
        if os.path.isfile(PREDICTION_HISTORY_FILE):
            history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
            history_count = len(history_df)
        
        return jsonify({
            'success': True,
            'metrics': {
                'r2_score': r2_score,
                'rmse': rmse,
                'predictions_stored': history_count
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/')
def home():
    """Render home page with project information"""
    try:
        # Get model metrics
        r2_score, rmse = get_model_metrics()
        total_predictions = len(df)  # In a real app, this would be tracked
        
        return render_template('home.html', 
                             r2_score=r2_score,
                             rmse=rmse,
                             total_predictions=total_predictions)
    except Exception as e:
        print(f"Error in home route: {str(e)}")
        return render_template('home.html',
                             r2_score=0.0,
                             rmse=0.0,
                             total_predictions=len(df))

@app.route('/analytics')
def analytics():
    """Render analytics page"""
    return render_template('analytics.html')

@app.route('/get_categories')
def get_categories():
    """Get list of unique categories"""
    categories = sorted(df['Category'].unique().tolist())
    return jsonify(categories)

@app.route('/get_items_by_category')
def get_items_by_category():
    """Get items for a specific category"""
    category = request.args.get('category')
    if not category:
        return jsonify({'error': 'Category not specified'}), 400
    
    items = sorted(df[df['Category'] == category]['Item_Name'].unique().tolist())
    return jsonify(items)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Starting prediction...")
        data = request.get_json()
        category = data.get('category')
        item_name = data.get('item_name')
        
        if not category or not item_name:
            return jsonify({
                'error': 'Please provide both category and item name',
                'success': False
            })

        # Get all items in the category for comparison
        category_data = df[df['Category'] == category].copy()
        if len(category_data) == 0:
            return jsonify({
                'error': 'Category not found in database',
                'success': False
            })
            
        # Get specific item data
        item_data = category_data[category_data['Item_Name'] == item_name].copy()
        if len(item_data) == 0:
            return jsonify({
                'error': 'Item not found in database',
                'success': False
            })

        try:
            # Calculate monthly averages for all items in category
            category_data['Month'] = pd.to_datetime(category_data['Timestamp']).dt.strftime('%B %Y')
            monthly_sales = category_data.groupby(['Month', 'Item_Name']).agg({
                'Item_Outlet_Sales': ['mean', 'count'],
                'Item_MRP': 'first'
            }).reset_index()
            
            # Rename columns for clarity
            monthly_sales.columns = ['Month', 'Item_Name', 'Average_Sales', 'Sales_Count', 'MRP']
            
            # Calculate units for each row
            monthly_sales['Average_Units'] = (monthly_sales['Average_Sales'] / monthly_sales['MRP']).round()
            
            # Sort by month (descending) and keep last 6 months
            months = sorted(monthly_sales['Month'].unique(), reverse=True)[:6]
            monthly_sales = monthly_sales[monthly_sales['Month'].isin(months)]
            
            # Get current and previous month for comparison
            current_month = months[0]
            previous_month = months[1] if len(months) > 1 else None
            
            # Prepare comparison data
            comparison_data = []
            unique_items = category_data['Item_Name'].unique()
            
            for curr_item in unique_items:
                curr_item_data = monthly_sales[monthly_sales['Item_Name'] == curr_item]
                current = curr_item_data[curr_item_data['Month'] == current_month].iloc[0] if len(curr_item_data[curr_item_data['Month'] == current_month]) > 0 else None
                previous = curr_item_data[curr_item_data['Month'] == previous_month].iloc[0] if previous_month and len(curr_item_data[curr_item_data['Month'] == previous_month]) > 0 else None
                
                # Get latest data for prediction
                latest_data = category_data[category_data['Item_Name'] == curr_item].iloc[-1:].copy()
                
                # Prepare features and predict
                X = prepare_input_features(latest_data)
                predicted_sales = float(model.predict(X)[0])
                predicted_units = round(predicted_sales / float(latest_data['Item_MRP'].iloc[0]))
                
                comparison_data.append({
                    'item_name': curr_item,
                    'current_month': current_month,
                    'current_sales': float(current['Average_Sales']) if current is not None else 0,
                    'current_units': int(current['Average_Units']) if current is not None else 0,
                    'previous_month': previous_month,
                    'previous_sales': float(previous['Average_Sales']) if previous is not None else 0,
                    'previous_units': int(previous['Average_Units']) if previous is not None else 0,
                    'predicted_sales': predicted_sales,
                    'predicted_units': predicted_units,
                    'growth': ((predicted_sales - (float(current['Average_Sales']) if current is not None else 0)) / 
                             (float(current['Average_Sales']) if current is not None and float(current['Average_Sales']) > 0 else 1) * 100)
                })
            
            # Sort comparison data by predicted sales
            comparison_data.sort(key=lambda x: x['predicted_sales'], reverse=True)
            
            # Format comparison data for response
            comparison_table = []
            for item in comparison_data:
                comparison_table.append({
                    'item_name': item['item_name'],
                    'current_month': item['current_month'],
                    'current_sales': format_currency(item['current_sales']),
                    'current_units': item['current_units'],
                    'previous_month': item['previous_month'],
                    'previous_sales': format_currency(item['previous_sales']),
                    'previous_units': item['previous_units'],
                    'predicted_sales': format_currency(item['predicted_sales']),
                    'predicted_units': item['predicted_units'],
                    'growth': f"{item['growth']:.1f}%"
                })
            
            # Get the target item's predictions
            target_item = next(item for item in comparison_data if item['item_name'] == item_name)
            predicted_demand = target_item['predicted_sales']
            predicted_units = target_item['predicted_units']
            
            # Calculate recent average for target item
            recent_data = item_data.sort_values('Timestamp', ascending=False).head(3)
            avg_demand = float(recent_data['Item_Outlet_Sales'].mean())
            avg_units = round(avg_demand / float(recent_data['Item_MRP'].iloc[0]))
            
            # Prepare bar graph data
            graph_data = {
                'items': [item['item_name'] for item in comparison_data],
                'current_sales': [item['current_sales'] for item in comparison_data],
                'current_units': [item['current_units'] for item in comparison_data],
                'predicted_sales': [item['predicted_sales'] for item in comparison_data],
                'predicted_units': [item['predicted_units'] for item in comparison_data]
            }
            
            # Format monthly table data
            monthly_table = []
            for _, row in monthly_sales.iterrows():
                monthly_table.append({
                    'month': row['Month'],
                    'item_name': row['Item_Name'],
                    'average_sales': format_currency(row['Average_Sales']),
                    'average_units': int(row['Average_Units']),
                    'num_sales': int(row['Sales_Count'])
                })
            
            response = {
                'success': True,
                'predicted_demand': format_currency(predicted_demand),
                'predicted_units': predicted_units,
                'average_demand': format_currency(avg_demand),
                'average_units': avg_units,
                'growth_potential': f"{target_item['growth']:.1f}%",
                'suggestion': get_suggestion(target_item['growth'], item_name),
                'monthly_table': monthly_table,
                'comparison_table': comparison_table,
                'graph_data': graph_data
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during prediction processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Error during prediction: {str(e)}',
                'success': False
            })
            
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        })

@app.route('/predict_timeline', methods=['POST'])
def predict_timeline():
    """Predict demand for different time periods"""
    try:
        data = request.get_json()
        timeline = data.get('timeline', '1_month')  # '1_month', '3_months', '1_year'
        
        # Prepare input features
        input_features = prepare_input_features(data)
        
        # Make base prediction
        base_prediction = model.predict([input_features])[0]
        
        # Adjust prediction based on timeline
        if timeline == '1_month':
            predictions = [base_prediction]
            months = [MONTHS[0]]
        elif timeline == '3_months':
            predictions = [base_prediction * (1 + i * 0.1) for i in range(3)]
            months = MONTHS[:3]
        else:  # 1_year
            predictions = [base_prediction * (1 + i * 0.15) for i in range(12)]
            months = MONTHS * 2  # Repeat the 6 months twice for a year
            
        # Convert predictions to integers and calculate MRP in INR
        predictions = [int(p) for p in predictions]
        mrp_inr = float(data.get('mrp', 0)) * USD_TO_INR
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'months': months,
            'mrp_inr': round(mrp_inr, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_suggestion(growth_potential, item_name):
    if growth_potential > 20:
        return f"Consider increasing stock levels for {item_name} as demand is expected to rise significantly."
    elif growth_potential > 10:
        return f"Moderate increase in demand expected for {item_name}. Monitor stock levels."
    elif growth_potential < -20:
        return f"Consider reducing stock levels for {item_name} as demand is expected to decrease significantly."
    elif growth_potential < -10:
        return f"Slight decrease in demand expected for {item_name}. Adjust inventory accordingly."
    else:
        return f"Demand for {item_name} is expected to remain stable. Maintain current stock levels."

def prepare_input_features(data):
    """Prepare input features for prediction"""
    global df, feature_columns
    
    try:
        print("Preparing input features...")
        print("Input data:", data)
        
        # Create DataFrame with numerical features
        numerical_features = ['Item_Weight', 'Item_MRP_INR', 'Shelf_Life_Days', 'Trending_Score', 
                            'Establishment_Year', 'Visibility']
        features = {}
        
        for col in numerical_features:
            features[col] = data.get(col, df[col].median())
        
        features_df = pd.DataFrame([features])
        
        # Add month encoding
        month_dummies = pd.get_dummies(pd.Series([data.get('Month', MONTHS[0])]), prefix='Month')
        features_df = pd.concat([features_df, month_dummies], axis=1)
        
        # Create dummy variables for categorical features
        categorical_features = {
            'Category': ['Dairy Products', 'Fruits and Vegetables', 'Grains and Cereals', 'Packaged Snacks'],
            'Seasonal_Demand': ['High', 'Low', 'Medium'],
            'Packaging_Type': ['Loose', 'Packaged'],
            'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
            'Fat_Content': ['Full Fat', 'Low Fat', 'Regular']
        }
        
        for feature, values in categorical_features.items():
            dummies = pd.get_dummies(pd.Series([data.get(feature, values[0])]), prefix=feature)
            features_df = pd.concat([features_df, dummies], axis=1)
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Select only the columns used during training
        features_df = features_df[feature_columns]
        
        print("Feature columns:", feature_columns)
        print("Prepared features shape:", features_df.shape)
        
        return features_df
        
    except Exception as e:
        print(f"Error in prepare_input_features: {str(e)}")
        raise

def get_model_metrics():
    """Get model performance metrics"""
    global model, df, feature_columns
    
    try:
        # Prepare features and target
        X = prepare_input_features(df.iloc[0])  # Get feature columns
        y = df['Item_Outlet_Sales']
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = model.score(X, y)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        return r2, rmse
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return 0.0, 0.0

if __name__ == '__main__':
    app.run(debug=True)
