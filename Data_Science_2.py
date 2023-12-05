# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:15:09 2023

@author: Fayssal
"""
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew, shapiro
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

# Importation des fichiers xlsx
stock_price_data = pd.read_excel("C://Users/Fayssal/Desktop/Projet Applied Data Science/20230214 - SP 500 prices.xlsx", header=3, skiprows=[4,5]) # On prend la 4ème ligne pour les noms de colonnes et on saute les lignes 5 et 6.
all_esg_sheets = pd.read_excel("C://Users/Fayssal/Desktop/Projet Applied Data Science/ESG-Sustainalitics.xlsx", sheet_name=None, header=3, skiprows=[4,5])
# 'Sustainalytics esg risk score':
sustainalytics_data = all_esg_sheets['Sustainalitics']

# Prétraitement des données
stock_price_data.set_index(stock_price_data.columns[0], inplace=True)
long_format = stock_price_data.stack().reset_index()
long_format.columns = ['Date', 'Ticker', 'Last Price']


non_numeric_rows = long_format[~long_format['Last Price'].apply(lambda x: isinstance(x, (int, float)))]

long_format['Last Price'] = pd.to_numeric(long_format['Last Price'], errors='coerce')

# Volatility calculation
long_format['Return'] = long_format.groupby('Ticker')['Last Price'].pct_change()
long_format['Volatility'] = long_format.groupby('Ticker')['Return'].rolling(window=30).std().reset_index(0, drop=True)


def compute_drawdowns(returns):
    cumulative = returns.cummax()
    drawdown = (returns - cumulative) / cumulative
    return drawdown

long_format['Drawdown'] = long_format.groupby('Ticker')['Return'].apply(compute_drawdowns)

def compute_skewness(group):
    return group.dropna().skew()

long_format['Skewness'] = long_format.groupby('Ticker')['Return'].transform(compute_skewness)

long_format.dropna(inplace=True)

# Prétraitement des données pour sustainalytics_data
sustainalytics_data.set_index(sustainalytics_data.columns[0], inplace=True)
long_format_sustainalytics = sustainalytics_data.stack().reset_index()
long_format_sustainalytics.columns = ['Date', 'Ticker', 'ESG Score']

non_numeric_rows_sustainalytics = long_format_sustainalytics[~long_format_sustainalytics['ESG Score'].apply(lambda x: isinstance(x, (int, float)))]

print(non_numeric_rows_sustainalytics)
long_format_sustainalytics['ESG Score'] = pd.to_numeric(long_format_sustainalytics['ESG Score'], errors='coerce')

long_format_sustainalytics.dropna(inplace=True)

# Merging with ESG scores
merged_data = pd.merge(long_format, long_format_sustainalytics, on=['Date', 'Ticker'], how='left')


# 1. EDA:

# a. Distributions:

# Histogramme de la volatilité
plt.figure(figsize=(10,5))
plt.title('Distribution de la Volatilité')
sns.histplot(long_format['Volatility'], kde=True)
plt.show()

# Histogramme de l'ESG score
plt.figure(figsize=(10,5))
plt.title('Distribution du Score ESG')
sns.histplot(merged_data['ESG Score'], kde=True)
plt.show()

# b. Scatter plot entre l'ESG score et la volatilité
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score ESG et la Volatilité')
sns.scatterplot(data=merged_data, x='ESG Score', y='Volatility')
plt.xlabel('Score ESG')
plt.ylabel('Volatilité')
plt.show()

# c. Coefficient de corrélation
correlation = merged_data['ESG Score'].corr(merged_data['Volatility'])
print(f"Le coefficient de corrélation entre le Score ESG et la Volatilité est de : {correlation:.2f}")


# 1. Multicolinéarité
# Matrice de corrélation
correlation_matrix = merged_data[['Volatility', 'Drawdown', 'Skewness']].corr()
print(correlation_matrix)


from sklearn.preprocessing import StandardScaler

merged_data = merged_data.dropna(subset=['Volatility', 'Drawdown', 'Skewness', 'ESG Score'])

# 2. Centrez et normalisez les variables explicatives.
scaler = StandardScaler()
merged_data[['Volatility', 'Drawdown', 'Skewness']] = scaler.fit_transform(merged_data[['Volatility', 'Drawdown', 'Skewness']])

# 3. Régression linéaire
X = sm.add_constant(merged_data[['Volatility', 'Drawdown', 'Skewness']])
y = merged_data['ESG Score']
model = sm.OLS(y, X).fit()
residuals = model.resid

# Loop through each predictor and plot
# Visualisation des relations entre chaque prédicteur et la réponse sans ligne de régression
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Loop through each predictor and plot
predictors = ['Volatility', 'Drawdown', 'Skewness']
for i, predictor in enumerate(predictors):
    axes[i].scatter(merged_data[predictor], merged_data['ESG Score'], alpha=0.5)
    axes[i].set_title(f'ESG Score vs {predictor}')
    axes[i].set_xlabel(predictor)
    axes[i].set_ylabel('ESG Score')

plt.tight_layout()
plt.show()

# QQ-Plot
qqplot(residuals, line='s')
plt.show()

# Test de Shapiro-Wilk
stat, p = shapiro(residuals)
print(f'Statistique de Shapiro-Wilk: {stat}, p-valeur: {p}')

# 3. Hétéroscédasticité
# Graphique des résidus
plt.scatter(model.predict(), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Graphique des résidus')
plt.show()

# Test de Breusch-Pagan
bp_test = het_breuschpagan(residuals, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))

# 4. Outliers
# Boxplots
features = ['Volatility', 'Drawdown', 'Skewness', 'ESG Score']
for feature in features:
    sns.boxplot(merged_data[feature])
    plt.title(f'Boxplot pour {feature}')
    plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Remplir les valeurs manquantes de X avec la moyenne de chaque colonne
X.fillna(X.mean(), inplace=True)

# Si y est une série (une seule colonne), remplissez-la avec sa propre moyenne
y.fillna(y.mean(), inplace=True)

# RF
rf_cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
rf_cv_mean = -rf_cv_scores.mean()
# Afficher les scores

# Lasso
lasso_cv_scores = cross_val_score(Lasso(alpha=0.1), X, y, cv=5, scoring='neg_mean_squared_error')
lasso_cv_mean = -lasso_cv_scores.mean()

# Ridge
ridge_cv_scores = cross_val_score(Ridge(alpha=0.1), X, y, cv=5, scoring='neg_mean_squared_error')
ridge_cv_mean = -ridge_cv_scores.mean()



# XGBoost
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=42)
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
xgb_cv_mean = -xgb_cv_scores.mean()

print("Scores for each fold:", -rf_cv_scores)
print("Moyenne des scores (MSE):", rf_cv_mean)

print(f"MSE Lasso (CV): {lasso_cv_mean}")
print(f"MSE Ridge (CV): {ridge_cv_mean}")

print("Scores for each fold (XGBoost):", -xgb_cv_scores)
print("MSE XGBoost:", xgb_cv_mean)


"""
BBG
"""
bbg_data = all_esg_sheets['BBG']

# Mise en forme longue
bbg_data.set_index(bbg_data.columns[0], inplace=True)
long_format_bbg = bbg_data.stack().reset_index()
long_format_bbg.columns = ['Date', 'Ticker', 'BBG Score']

# Vérification des lignes non numériques
non_numeric_rows_bbg = long_format_bbg[~long_format_bbg['BBG Score'].apply(lambda x: isinstance(x, (int, float)))]

# Conversion forcée en numérique
long_format_bbg['BBG Score'] = pd.to_numeric(long_format_bbg['BBG Score'], errors='coerce')

# Élimination des valeurs manquantes
long_format_bbg.dropna(inplace=True)

# Fusion avec BBG scores
merged_data = pd.merge(merged_data, long_format_bbg, on=['Date', 'Ticker'], how='left')

# Histogramme du score BBG
plt.figure(figsize=(10,5))
plt.title('Distribution du Score BBG')
sns.histplot(merged_data['BBG Score'], kde=True)
plt.show()

# Scatter plot entre le score BBG et la volatilité
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score BBG et la Volatilité')
sns.scatterplot(data=merged_data, x='BBG Score', y='Volatility')
plt.xlabel('Score BBG')
plt.ylabel('Volatilité')
plt.show()

# Scatter plot entre le score BBG et le Skewness
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score BBG et le Skewness')
sns.scatterplot(data=merged_data, x='BBG Score', y='Skewness')
plt.xlabel('Score BBG')
plt.ylabel('Skewness')
plt.show()

# Scatter plot entre le score BBG et le Drawdown
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score BBG et le Drawdown')
sns.scatterplot(data=merged_data, x='BBG Score', y='Drawdown')
plt.xlabel('Score BBG')
plt.ylabel('Drawdown')
plt.show()

# c. Coefficient de corrélation
correlation = merged_data['BBG Score'].corr(merged_data['Volatility'])
print(f"Le coefficient de corrélation entre le Score BBG et la Volatilité est de : {correlation:.2f}")

# 1. Multicolinéarité
# Matrice de corrélation
correlation_matrix = merged_data[['Volatility', 'Drawdown', 'Skewness']].corr()
print(correlation_matrix)

from sklearn.preprocessing import StandardScaler

# 1. Assurez-vous qu'il n'y a pas de valeurs manquantes.
merged_data = merged_data.dropna(subset=['Volatility', 'Drawdown', 'Skewness', 'BBG Score'])

# 2. Centrez et normalisez les variables explicatives.
scaler = StandardScaler()
merged_data[['Volatility', 'Drawdown', 'Skewness']] = scaler.fit_transform(merged_data[['Volatility', 'Drawdown', 'Skewness']])

# 3. Régression linéaire
X = sm.add_constant(merged_data[['Volatility', 'Drawdown', 'Skewness']])
y = merged_data['BBG Score']
model = sm.OLS(y, X).fit()
residuals = model.resid

# Visualisation des relations entre chaque prédicteur et la réponse sans ligne de régression
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Loop through each predictor and plot
predictors = ['Volatility', 'Drawdown', 'Skewness']
for i, predictor in enumerate(predictors):
    axes[i].scatter(merged_data[predictor], merged_data['BBG Score'], alpha=0.5)
    axes[i].set_title(f'BBG Score vs {predictor}')
    axes[i].set_xlabel(predictor)
    axes[i].set_ylabel('BBG Score')

plt.tight_layout()
plt.show()

# QQ-Plot
qqplot(residuals, line='s')
plt.show()

# Test de Shapiro-Wilk
stat, p = shapiro(residuals)
print(f'Statistique de Shapiro-Wilk: {stat}, p-valeur: {p}')

# 3. Hétéroscédasticité
# Graphique des résidus
plt.scatter(model.predict(), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Graphique des résidus')
plt.show()

# Test de Breusch-Pagan
bp_test = het_breuschpagan(residuals, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))

# 4. Outliers
# Boxplots
features = ['Volatility', 'Drawdown', 'Skewness', 'BBG Score']
for feature in features:
    sns.boxplot(merged_data[feature])
    plt.title(f'Boxplot pour {feature}')
    plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Remplir les valeurs manquantes de X avec la moyenne de chaque colonne
X.fillna(X.mean(), inplace=True)

# Si y est une série (une seule colonne), remplissez-la avec sa propre moyenne
y.fillna(y.mean(), inplace=True)

# RF
rf_cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
rf_cv_mean = -rf_cv_scores.mean()

# Lasso
lasso_cv_scores = cross_val_score(Lasso(alpha=0.1), X, y, cv=5, scoring='neg_mean_squared_error')
lasso_cv_mean = -lasso_cv_scores.mean()

# Ridge
ridge_cv_scores = cross_val_score(Ridge(alpha=0.1), X, y, cv=5, scoring='neg_mean_squared_error')
ridge_cv_mean = -ridge_cv_scores.mean()

print("Scores for each fold:", -rf_cv_scores)
print("Mean Squared Error:", rf_cv_mean)

print(f"MSE Lasso (CV): {lasso_cv_mean}")
print(f"MSE Ridge (CV): {ridge_cv_mean}")
# XGBoost
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=42)
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
xgb_cv_mean = -xgb_cv_scores.mean()

print("Scores for each fold (XGBoost):", -xgb_cv_scores)
print("MSE XGBoost:", xgb_cv_mean)

robeco_data = all_esg_sheets['RobecoSAM']

robeco_data.set_index(robeco_data.columns[0], inplace=True)
long_format_robeco = robeco_data.stack().reset_index()
long_format_robeco.columns = ['Date', 'Ticker', 'RobecoSAM Score']

non_numeric_rows_robeco = long_format_robeco[~long_format_robeco['RobecoSAM Score'].apply(lambda x: isinstance(x, (int, float)))]

long_format_robeco['RobecoSAM Score'] = pd.to_numeric(long_format_robeco['RobecoSAM Score'], errors='coerce')

long_format_robeco.dropna(inplace=True)

merged_data = pd.merge(merged_data, long_format_robeco, on=['Date', 'Ticker'], how='left')


# Histogramme du score RobecoSAM
plt.figure(figsize=(10,5))
plt.title('Distribution du Score RobecoSAM')
sns.histplot(merged_data['RobecoSAM Score'], kde=True)
plt.show()

# Scatter plot entre le score RobecoSAM et la volatilité
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score RobecoSAM et la Volatilité')
sns.scatterplot(data=merged_data, x='RobecoSAM Score', y='Volatility')
plt.xlabel('Score RobecoSAM')
plt.ylabel('Volatilité')
plt.show()

# Scatter plot entre le score RobecoSAM et le Skewness
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score RobecoSAM et le Skewness')
sns.scatterplot(data=merged_data, x='RobecoSAM Score', y='Skewness')
plt.xlabel('Score RobecoSAM')
plt.ylabel('Skewness')
plt.show()

# Scatter plot entre le score RobecoSAM et le Drawdown
plt.figure(figsize=(10,5))
plt.title('Relation entre le Score RobecoSAM et le Drawdown')
sns.scatterplot(data=merged_data, x='RobecoSAM Score', y='Drawdown')
plt.xlabel('Score RobecoSAM')
plt.ylabel('Drawdown')
plt.show()

# c. Coefficient de corrélation
correlation = merged_data['RobecoSAM Score'].corr(merged_data['Volatility'])
print(f"Le coefficient de corrélation entre le Score RobecoSAM et la Volatilité est de : {correlation:.2f}")

# 1. Multicolinéarité
# Matrice de corrélation
correlation_matrix = merged_data[['Volatility', 'Drawdown', 'Skewness']].corr()
print(correlation_matrix)

# Assurez-vous qu'il n'y a pas de valeurs manquantes.
merged_data = merged_data.dropna(subset=['Volatility', 'Drawdown', 'Skewness', 'RobecoSAM Score'])

# Centrez et normalisez les variables explicatives.
scaler = StandardScaler()
merged_data[['Volatility', 'Drawdown', 'Skewness']] = scaler.fit_transform(merged_data[['Volatility', 'Drawdown', 'Skewness']])

# Régression linéaire
X = sm.add_constant(merged_data[['Volatility', 'Drawdown', 'Skewness']])
y = merged_data['RobecoSAM Score']
model = sm.OLS(y, X).fit()
residuals = model.resid

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

predictors = ['Volatility', 'Drawdown', 'Skewness']
for i, predictor in enumerate(predictors):
    axes[i].scatter(merged_data[predictor], merged_data['RobecoSAM Score'], alpha=0.5)
    axes[i].set_title(f'RobecoSAM Score vs {predictor}')
    axes[i].set_xlabel(predictor)
    axes[i].set_ylabel('RobecoSAM Score')

plt.tight_layout()
plt.show()

# QQ-Plot
qqplot(residuals, line='s')
plt.show()

# Test de Shapiro-Wilk
stat, p = shapiro(residuals)
print(f'Statistique de Shapiro-Wilk: {stat}, p-valeur: {p}')

# 3. Hétéroscédasticité
# Graphique des résidus
plt.scatter(model.predict(), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Graphique des résidus')
plt.show()

# Test de Breusch-Pagan
bp_test = het_breuschpagan(residuals, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))

# 4. Outliers
features = ['Volatility', 'Drawdown', 'Skewness', 'RobecoSAM Score']
for feature in features:
    sns.boxplot(merged_data[feature])
    plt.title(f'Boxplot pour {feature}')
    plt.show()

# Valeurs manquantes de X avec la moyenne de chaque colonne
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# RF
rf_cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
rf_cv_mean = -rf_cv_scores.mean()
print("Scores for each fold:", -rf_cv_scores)
print("MSE:", rf_cv_mean)

# Lasso
lasso_cv_scores = cross_val_score(Lasso(alpha=0.1), X, y, cv=5, scoring='neg_mean_squared_error')
lasso_cv_mean = -lasso_cv_scores.mean()

# Ridge
ridge_cv_scores = cross_val_score(Ridge(alpha=0.1), X, y, cv=5, scoring='neg_mean_squared_error')
ridge_cv_mean = -ridge_cv_scores.mean()

print(f"MSE (CV): {lasso_cv_mean}")
print(f"MSE (CV): {ridge_cv_mean}")


# XGBoost
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=42)
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
xgb_cv_mean = -xgb_cv_scores.mean()

print("Scores for each fold (XGBoost):", -xgb_cv_scores)
print("MSE XGBoost:", xgb_cv_mean)


