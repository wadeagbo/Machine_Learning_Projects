📁 Project 1: Carbon Monoxide (CO) Prediction Using Air Quality Data
🎯 Project Aim:
To predict CO levels using the UCI Air Quality dataset, applying time-series and machine learning techniques for environmental health forecasting.

✅ What Was Done:

Data Preprocessing: Handled missing/invalid values, formatted timestamps, and filled gaps using forward fill.
Feature Engineering: Generated time-based features (Hour, DayOfWeek, Month) to capture seasonal trends.
Modeling: Trained an XGBRegressor after applying StandardScaler.
Evaluation: Used Mean Squared Error (MSE) to assess model performance; forecasted CO levels for the next 24 hours.
Visualization: Plotted actual vs predicted CO concentrations.
💡 Suggestions:

Include weather or pollutant variables to improve feature richness.
Integrate real-time data and apply hyperparameter tuning for better accuracy.
📁 Project 2: Wind Turbine Performance Analysis and Prediction
🎯 Project Aim:
To classify wind turbine operational status by comparing measured and theoretical power outputs—identifying underperformance for optimization and maintenance.

✅ What Was Done:

Data Cleaning: Removed anomalies, filled missing values, and ensured consistent formatting.
Feature Engineering: Created lag features (Wind_Lag1, Wind_Lag7) to capture temporal patterns.
Modeling: Used multiple classifiers: Logistic Regression, Random Forest, Gradient Boosting, SVM, and KNN.
Evaluation: Compared models using accuracy, precision, recall, F1-score, and confusion matrices.
Tuning & Interpretation: Applied grid search for optimal parameters; analyzed feature importance.
💡 Suggestions:

Random Forest and Gradient Boosting yielded the best results.
Adding environmental data or turbine history could boost predictive power.
Deployment for real-time turbine monitoring is a promising next step.
📁 Project 3: Hyperspectral Image Denoising and Analysis (Indian Pines Dataset)
🎯 Project Aim:
To process and denoise hyperspectral image data using PCA, Total Variation (TV), and deep learning techniques, enhancing data quality for classification or further analysis.

✅ What Was Done:

Data Loading & Visualization: Loaded Indian Pines dataset; visualized individual bands and RGB composites.
Preprocessing: Flattened the 3D hyperspectral cube and filtered out unlabeled pixels for focused analysis.
Denoising with PCA: Applied Principal Component Analysis to reduce noise and dimensionality.
Autoencoder (PyTorch): Built and trained a denoising autoencoder on synthetic noisy spectral data.
TV Denoising: Applied Total Variation filtering to a noisy band using skimage.
💡 Suggestions:

