import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Part 1
# Task 1
data = pd.read_csv("LifeExpectancy.csv")
y = data["Life expectancy "].fillna(data["Life expectancy "].mean())


# Task 2: See the model performance based on one feature
# Define features and initialize models
features = {
    "GDP": "GDP",
    "Total expenditure": "Total expenditure",
    "Alcohol": "Alcohol"
}

models = {}
scalers = {}
results = []

# Set a fixed random seed for consistent splits
random_seed = 42

for feature_name, col_name in features.items():
    # Preprocess feature
    X = data[[col_name]].fillna(data[col_name].mean())

    # Split into training/test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Standardize feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Get coefficients in original scale
    slope_scaled = model.coef_[0]
    intercept = model.intercept_
    mean_feature = scaler.mean_[0]
    std_feature = scaler.scale_[0]
    slope_original = slope_scaled / std_feature
    intercept_original = intercept - (slope_scaled * mean_feature / std_feature)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append({
        "Feature": feature_name,
        "Slope": slope_original,
        "Intercept": intercept_original,
        "MSE": mse,
        "R²" : r2
    })

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_train, y_train,
        color="blue", alpha=0.5, label="Training Data"
    )

    x_min = X_train.min().iloc[0]  # Use .iloc[0] to avoid FutureWarning
    x_max = X_train.max().iloc[0]

    # Create DataFrame to preserve feature name
    x_line = pd.DataFrame(
        np.linspace(x_min, x_max, 100).reshape(-1, 1),
        columns=X_train.columns  # Match feature name from training data
    )

    x_line_scaled = scaler.transform(x_line)  # Now has valid feature names
    y_line = model.predict(x_line_scaled)

    plt.plot(
        x_line, y_line,
        color="red", linewidth=2,
        label=f"Regression Line: $y = {slope_original:.2f}x + {intercept_original:.2f}$"
    )

    plt.text(
        0.05, 0.85,
        f"$R^2 = {r2:.2f}$\nMSE = {mse:.2f}",  # Display metrics
        transform=plt.gca().transAxes,  # Use axis coordinates
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)  # White background box
    )

    plt.xlabel(f"{feature_name} (Original Scale)")  # X-axis label
    plt.ylabel("Life Expectancy")  # Y-axis label
    plt.title(f"{feature_name} vs Life Expectancy")  # Plot title

    plt.legend()  # Show labels for data and regression line
    plt.grid(True)  # Add gridlines for readability
    plt.show()

# Part 2: Create a model based on 4 best features and compare its performance with other models, which are trained only
# on 1 feature
# Task 2
# Select features and target for my model
X_features = data[["Adult Mortality", "percentage expenditure", "GDP", "Schooling"]]
y_target = data["Life expectancy "]

# Preprocess data
X_features = X_features.fillna(X_features.mean())
y_target = y_target.fillna(y_target.mean())

# Splitting into train/test datasets
X_features_train, X_features_test, y_target_train, y_target_test = train_test_split(X_features,
                                                                                    y_target,
                                                                                    test_size=0.2,
                                                                                    random_state=42)
# Standardization
multiple_scaler = StandardScaler()
X_features_train_scaled = multiple_scaler.fit_transform(X_features_train)
X_features_test_scaled = multiple_scaler.transform(X_features_test)

# Train model
optimized_model = LinearRegression()
optimized_model.fit(X_features_train_scaled, y_target_train)

# Predict and evaluate
y_optimized_model_pred = optimized_model.predict(X_features_test_scaled)
mse_optimized = mean_squared_error(y_target_test, y_optimized_model_pred)
r2_optimized = r2_score(y_target_test, y_optimized_model_pred)

print(f"MSE for a model with multiple features: {mse_optimized:.2f}")
print(f"R² for a model with multiple features: {r2_optimized:.2f}")
print(f"Standard deviation for a model with multiple features: {multiple_scaler.scale_}")
print(f"Mean for a model with multiple features: {multiple_scaler.mean_.round(2)}")



# Original coefficients
original_slopes = [(optimized_model.coef_[x]/ multiple_scaler.scale_[x]).round(2) for x in range(4)]
optimized_intercept = optimized_model.intercept_ - np.sum(optimized_model.coef_ * (multiple_scaler.mean_ / multiple_scaler.scale_))
print(original_slopes)
print(optimized_intercept)

coefficients = pd.DataFrame({
    "Feature": X_features.columns,
    "Coefficient": original_slopes
})
print(f"Feature coefficients and intercept: {coefficients}\nIntercept: {optimized_intercept}")


