📁 Project 1: Carbon Monoxide (CO) Prediction Using Air Quality Data

🎯 Project Aim
Predict CO levels using the UCI Air Quality dataset by applying time-series analysis and machine learning for environmental health forecasting.

✅ What Was Done
Data Preprocessing: Handled missing/invalid values, formatted timestamps, and filled gaps using forward fill.
Feature Engineering: Created time-based features (Hour, DayOfWeek, Month) to capture seasonal trends.
Modeling: Trained an XGBRegressor after scaling data with StandardScaler.
Evaluation: Assessed performance using Mean Squared Error (MSE); forecasted CO levels for the next 24 hours.
Visualization: Plotted actual vs predicted CO concentrations.
💡 Suggestions
Include weather or other pollutant variables to enrich features.
Incorporate real-time data and hyperparameter tuning to improve accuracy.
📁 Project 2: Wind Turbine Performance Analysis and Prediction

🎯 Project Aim
Classify wind turbine operational status by comparing measured and theoretical power output to detect underperformance and aid optimization and maintenance.

✅ What Was Done
Data Cleaning: Removed anomalies, handled missing values, ensured consistent data formatting.
Feature Engineering: Created lag features (e.g., Wind_Lag1, Wind_Lag7) to capture temporal dependencies.
Modeling: Tested multiple classifiers including Logistic Regression, Random Forest, Gradient Boosting, SVM, and KNN.
Evaluation: Measured model accuracy, precision, recall, F1-score, and confusion matrices.
Tuning & Interpretation: Optimized Random Forest using grid search and analyzed feature importance.
💡 Suggestions
Random Forest and Gradient Boosting performed best but need further optimization.
Adding turbine maintenance history and environmental data could enhance predictions.
Real-time prediction and cloud deployment would increase practical utility.
📁 Project 3: Hyperspectral Image Denoising and Analysis (Indian Pines Dataset)

🎯 Project Aim
Process and denoise hyperspectral image data using PCA, Total Variation, and deep learning methods to improve data quality for analysis and classification.

✅ What Was Done
Data Loading & Visualization: Loaded Indian Pines dataset; visualized individual spectral bands and false-color RGB composites.
Preprocessing: Flattened the 3D hyperspectral cube and filtered out unlabeled pixels for focused analysis.
PCA Denoising: Applied Principal Component Analysis for noise reduction and dimensionality compression.
Autoencoder (PyTorch): Built and trained a denoising autoencoder on synthetic noisy spectral data.
Total Variation Denoising: Applied TV filtering to a noisy spectral band using skimage.

💡 Suggestions for Improvement
Explore additional hyperspectral datasets to test the generalizability of denoising methods.
Experiment with advanced denoising techniques such as Variational Autoencoders (VAEs) or GAN-based models for potentially better results.
Compare PCA with alternative dimensionality reduction methods like t-SNE, UMAP, or Independent Component Analysis (ICA) to enhance feature extraction.
Integrate denoising with downstream classification workflows to evaluate impact on classification accuracy.
Use realistic sensor noise models to simulate training data for more robust autoencoder training.
Perform hyperparameter tuning on autoencoder architecture and PCA components to optimize denoising performance.
Develop interactive visualizations for exploring spectral bands and visualizing denoising effects effectively.
Add detailed documentation and reusable scripts covering data preprocessing, denoising, and evaluation for improved reproducibility.
