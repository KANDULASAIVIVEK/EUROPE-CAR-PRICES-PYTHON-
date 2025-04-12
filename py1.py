import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('car_prices.csv')

print("Dataset Columns:", df.columns.tolist())

# Data Preprocessing
def preprocess_data(df):
    df_clean = df.dropna().copy()
    column_mapping = {
        'brand': 'Brand',
        'model': 'Model',
        'year': 'Year',
        'kilometers_driven': 'Kilometers',
        'fuel_type': 'Fuel',
        'drivetrain': 'Drivetrain',
        'gearbox': 'Gearbox',
        'body_type': 'Body Type',
        'country': 'Country',
        'price': 'Price',
        'power': 'Power'
    }
    selected_cols = {}
    for key, col in column_mapping.items():
        if col in df_clean.columns:
            selected_cols[key] = col
        else:
            print(f"Warning: Column {col} not found for {key}")
    
    categorical_cols = [selected_cols.get(col) for col in ['brand', 'model', 'fuel_type', 'drivetrain', 'gearbox', 'body_type', 'country'] if col in selected_cols]
    for col in categorical_cols:
        unique_vals = df_clean[col].astype(str).unique()
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        df_clean[col] = df_clean[col].astype(str).map(val_to_int).astype(float)
    
    numeric_cols = [selected_cols.get(col) for col in ['year', 'kilometers_driven', 'price', 'power'] if col in selected_cols]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(float)
    
    return df_clean, selected_cols

df, selected_cols = preprocess_data(df)

# 1. Price Prediction Model
def train_price_prediction_model(df, selected_cols):
    features = [selected_cols.get(col) for col in ['brand', 'model', 'year', 'kilometers_driven', 'fuel_type', 'drivetrain'] if col in selected_cols]
    price_col = selected_cols.get('price', 'Price')
    
    if not features or price_col not in df.columns:
        print("Missing features or price column.")
        return None
    
    X = df[features]
    y = df[price_col]
    X = sm.add_constant(X.astype(float))
    y = y.astype(float)
    
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[indices[:train_size]], X.iloc[indices[train_size:]]
    y_train, y_test = y.iloc[indices[:train_size]], y.iloc[indices[train_size:]]
    
    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = model.rsquared
    
    print(f"Price Prediction Model - MSE: {mse:.2f}, R2: {r2:.2f}")
    print(model.summary())
    
    return model

price_model = train_price_prediction_model(df, selected_cols)

# 2. Gearbox Analysis
def gearbox_analysis(df, selected_cols):
    gearbox_col = selected_cols.get('gearbox')
    price_col = selected_cols.get('price')
    
    if gearbox_col and price_col:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=gearbox_col, y=price_col, data=df, palette='pastel')
        plt.title('Price Distribution by Gearbox Type')
        plt.show()

gearbox_analysis(df, selected_cols)

# 3. Fuel Type Impact
def fuel_type_analysis(df, selected_cols):
    fuel_col = selected_cols.get('fuel_type')
    price_col = selected_cols.get('price')
    year_col = selected_cols.get('year')
    
    if fuel_col and price_col:
        fuel_price = df.groupby(fuel_col)[price_col].mean().sort_values()
        print("\nAverage Price by Fuel Type:")
        print(fuel_price)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=fuel_price.index, y=fuel_price.values, palette='Set2')
        plt.title('Average Price by Fuel Type')
        plt.xticks(rotation=45)
        plt.show()
        
        # Changed scatter plot to line plot here
        if year_col:
            plt.figure(figsize=(10, 6))
            for fuel_type in df[fuel_col].unique():
                subset = df[df[fuel_col] == fuel_type]
                subset = subset.groupby(year_col)[price_col].mean().reset_index()
                plt.plot(subset[year_col], subset[price_col], label=f'Fuel {int(fuel_type)}')
            plt.title('Average Price over Years by Fuel Type')
            plt.xlabel('Year')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

fuel_type_analysis(df, selected_cols)

# 4. Country-wise Price Variation (Changed bar chart to pie chart)
def country_price_analysis(df, selected_cols):
    country_col = selected_cols.get('country')
    price_col = selected_cols.get('price')
    
    if country_col and price_col:
        country_price = df.groupby(country_col)[price_col].mean()
        print("\nAverage Price by Country:")
        print(country_price)
        
        plt.figure(figsize=(8, 8))
        plt.pie(country_price, labels=country_price.index, autopct='%1.1f%%', startangle=140)
        plt.title('Average Price Share by Country')
        plt.axis('equal')
        plt.show()

country_price_analysis(df, selected_cols)

# 5. Mileage Estimation
def mileage_estimation(df, selected_cols):
    fuel_col = selected_cols.get('fuel_type')
    power_col = selected_cols.get('power')
    km_col = selected_cols.get('kilometers_driven')
    
    if fuel_col and power_col and km_col:
        base_efficiency = {0: 12, 1: 15, 2: 10, 3: 20, 4: 18}
        df['estimated_km_per_liter'] = df.apply(
            lambda row: base_efficiency.get(row[fuel_col], 12) * (100 / max(row[power_col], 1)) * (100000 / max(row[km_col], 1)),
            axis=1
        )
        
        print("\nSample Mileage Estimations:")
        print(df[[fuel_col, power_col, km_col, 'estimated_km_per_liter']].head())
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=fuel_col, y='estimated_km_per_liter', data=df, palette='Accent')
        plt.title('Estimated Mileage by Fuel Type')
        plt.show()

mileage_estimation(df, selected_cols)

# 6. Body Type and Price Correlation
def body_type_analysis(df, selected_cols):
    body_col = selected_cols.get('body_type')
    price_col = selected_cols.get('price')
    year_col = selected_cols.get('year')
    
    if body_col and price_col:
        body_price = df.groupby(body_col)[price_col].mean().sort_values()
        print("\nAverage Price by Body Type:")
        print(body_price)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=body_price.index, y=body_price.values, palette='spring')
        plt.title('Average Price by Body Type')
        plt.xticks(rotation=45)
        plt.show()
        
        if year_col:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=year_col, y=price_col, hue=body_col, data=df)
            plt.title('Price vs Year by Body Type')
            plt.show()

body_type_analysis(df, selected_cols)

# Additional Visualizations

def plot_histograms(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    numeric_cols.hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Numeric Features')
    plt.show()

def plot_price_box(df, selected_cols):
    price_col = selected_cols.get('price')
    if price_col:
        plt.figure(figsize=(8, 5))
        sns.boxplot(y=df[price_col], color='tomato')
        plt.title("Box Plot of Car Prices")
        plt.ylabel("Price")
        plt.show()

def plot_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

plot_histograms(df)
plot_price_box(df, selected_cols)
plot_heatmap(df)
