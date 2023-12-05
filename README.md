# ESG Score Impact on Stock Prices Analysis

## Project Overview
This study aims to explore the potential relationships and impacts of ESG (Environmental, Social, and Governance) scores on stock prices in the American market. Utilizing data from two Excel files, the analysis includes various ESG scores for each stock in the S&P 500 on every trading date. One file contains Sustainalytics data with three different types of ESG scores, while the other file provides corresponding stock prices for the S&P 500 stocks covering the same periods.

## Data Description
- **Sustainalytics Data**: Contains three sheets, each with a unique type of ESG score for each S&P 500 stock on each trading date.
- **Stock Price Data**: Includes the prices of S&P 500 stocks corresponding to the dates and stocks in the Sustainalytics file.

## Analysis Workflow
1. **Data Importation and Preprocessing**:
   - Importation of ESG scores and stock prices from Excel files.
   - Preprocessing to format and clean the data for analysis.

2. **Exploratory Data Analysis (EDA)**:
   - Visualization of volatility and ESG score distributions.
   - Scatter plots to explore relationships between ESG scores and stock volatility.
   - Correlation coefficient calculation between ESG scores and volatility.

3. **Multicollinearity Check**:
   - Correlation matrix creation to identify multicollinearity among variables.

4. **Linear Regression Analysis**:
   - Standardizing and normalizing explanatory variables.
   - Running linear regression to understand the impact of variables on ESG scores.
   - Residuals analysis including QQ-Plot and Shapiro-Wilk test.

5. **Heteroscedasticity Test**:
   - Breusch-Pagan test to check for heteroscedasticity in the regression model.

6. **Outlier Detection**:
   - Boxplot visualization for key features.

7. **Model Evaluation**:
   - Using various regression models including RandomForest, Lasso, Ridge, and XGBoost.
   - Cross-validation to evaluate model performance based on Mean Squared Error (MSE).

## Libraries Used
- `xgboost`
- `matplotlib`
- `seaborn`
- `pandas`
- `scipy`
- `statsmodels`
- `sklearn`

## Running the Project
Ensure Python and the above-mentioned libraries are installed. The project script is a `.py` file developed in Spyder. To run the analysis, open and execute the script in a Python environment like Spyder or any other IDE that supports Python scripts, do not forget to change file-path when importing the data from both Sustainalytics and SP500 prices.

## Note
This project is for educational purposes and demonstrates data analysis and visualization techniques in Python. Please refer to the accompanying PDF for detailed study results.

