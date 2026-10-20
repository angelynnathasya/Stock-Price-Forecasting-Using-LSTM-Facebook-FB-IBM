# Stock Price Time Series Forecasting with LSTM: Facebook (FB) and IBM

Time series forecasting of historical Facebook (FB) and IBM stock closing prices using Persistence Baselines, Baseline LSTM, and Modified LSTM. Includes exploratory data analysis, preprocessing, chronological train-validation-test splitting, model training, visualizations, and evaluation with RMSE, MAE, and MAPE.

---

## Project Overview

This project explores time series forecasting using historical stock closing prices for Facebook (FB) and IBM. The workflow includes exploratory data analysis (EDA), data preprocessing, sequence generation, baseline modeling, LSTM modeling, and performance comparison.

The project follows the assignment constraints:
- **Input window size:** 5 observations
- **Forecast horizon:** 1 observation
- **Training/validation split:** 90% training and 10% validation, in chronological order
- **Test set:** the final year of observations
- **Input feature:** `Close`
- **Target:** next observation's closing price

The notebook compares three approaches:
1. **Persistence Baseline** — predicts the next closing price using the latest observed closing price.
2. **Baseline LSTM** — a simple LSTM neural network.
3. **Modified LSTM** — a modified LSTM architecture and training configuration.

## Objectives

- Explore the historical behavior and distribution of FB and IBM closing prices.
- Prepare chronological time series data for supervised learning.
- Build input-output sequences with a fixed window size and forecast horizon.
- Train and compare Persistence Baseline, Baseline LSTM, and Modified LSTM models.
- Evaluate forecasting errors using RMSE, MAE, and MAPE.
- Analyze whether the modified architecture improves on the baseline models.

## Dataset

The notebook uses two stock time series:
- **Facebook (FB):** historical closing prices from the provided dataset.
- **IBM:** historical closing prices from the provided dataset.

The notebook selects the `Date` and `Close` columns. `Date` is converted to datetime and used to preserve chronological order; `Close` is the forecasting feature and target.

## Methodology

### 1. Exploratory Data Analysis
The notebook explores:
- Statistical summaries
- Time series plots
- Closing-price distributions
- Boxplots and outlier inspection
- Trend analysis using a 30-observation moving average

### 2. Data Preprocessing
- Convert `Date` to datetime.
- Sort observations chronologically.
- Check missing values, duplicates, and data inconsistencies.
- Select `Date` and `Close`.
- Split the final year into the test set and use earlier observations for training.
- Scale closing prices with `MinMaxScaler`.

### 3. Sequence Generation
The project follows the required sequence configuration:

- **Window size = 5:** use five consecutive observations as input.
- **Horizon = 1:** predict the next observation.
- **Train/validation = 90%/10%:** split the training sequences chronologically, without shuffling.

### 4. Models

**Persistence Baseline**
- Predicts the next closing price using the most recent observed closing price.

**Baseline LSTM**
- One LSTM layer with 50 units
- ReLU activation
- One output neuron

**Modified LSTM**
- Uses a modified architecture and training configuration, including changes to LSTM units/layers, dropout, dense layers, optimizer/training settings, and early stopping as explored in the notebook.

The exact model configurations and experiment code are documented in the Jupyter notebook.

### 5. Evaluation
Predictions are converted back to the original price scale before calculating:
- **RMSE (Root Mean Squared Error):** penalizes larger prediction errors more strongly.
- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted prices.
- **MAPE (Mean Absolute Percentage Error):** average absolute percentage error.

For all three metrics, **lower values indicate smaller prediction errors**.

## Results

| Metric | Persistence FB | Persistence IBM | Baseline LSTM FB | Baseline LSTM IBM | Modified LSTM FB | Modified LSTM IBM |
|---|---:|---:|---:|---:|---:|---:|
| RMSE | 4.1735 | 2.6892 | 5.4912 | **2.6634** | 5.5691 | 3.5302 |
| MAE | **2.8427** | **1.7202** | 4.3111 | 1.7815 | 4.5636 | 2.2979 |
| MAPE (%) | **1.5483** | **1.3298** | 2.2657 | 1.3657 | 2.4155 | 1.7672 |

### Key Findings

- **Facebook (FB):** Persistence Baseline achieved the lowest RMSE, MAE, and MAPE among the three approaches.
- **IBM:** Baseline LSTM achieved the lowest RMSE, while Persistence Baseline achieved the lowest MAE and MAPE.
- **Modified LSTM:** The modified architecture did not outperform Baseline LSTM on the reported test metrics for either dataset.
- The results illustrate that a more complex neural network does not necessarily outperform a simple forecasting baseline.
- The Modified LSTM results suggest that the tested architecture and training settings may not be well suited to these datasets. This is an empirical finding, not proof that LSTM models are inherently unsuitable for stock forecasting.

## Requirements

The notebook uses Python and common data science libraries, including:
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Limitations

- The models use only historical closing prices and do not include other potentially informative variables such as Open, High, Low, or Volume.
- Stock prices are affected by many external factors that are not represented in this univariate dataset.
- Results depend on the chosen time period, preprocessing, model configuration, and random initialization.
- The Modified LSTM did not outperform the baseline models in the reported experiments.
- This project is intended for educational purposes and should not be interpreted as financial advice.

## Future Improvements

Subject to the assignment requirements, future work could include:
- Testing additional LSTM units, dropout rates, and learning rates while keeping window size 5 and horizon 1.
- Using `EarlyStopping` and `ReduceLROnPlateau` to manage training.
- Fixing random seeds and repeating experiments to assess result variability.
- Comparing models using the same data split and evaluation procedure.
- Adding features such as Open, High, Low, and Volume only if permitted and available at prediction time.


Data Science student — BINUS University

---

*This project was developed as an academic time series forecasting exercise.*
