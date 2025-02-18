import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Configurações de visualização
sns.set(style="whitegrid")
%matplotlib inline


# Caminhos para os arquivos CSV
r_csv_path = 'R_values.csv'        # Atualize com o caminho correto se necessário
spo2_csv_path = 'SpO2_real.csv'    # Atualize com o caminho correto se necessário

# Carregar os dados, lendo apenas as primeiras 10 linhas
r_df = pd.read_csv(r_csv_path, nrows=10)
spo2_df = pd.read_csv(spo2_csv_path, nrows=10)

# Verificar os dados carregados
print("Dados de R:")
print(r_df.head())

print("\nDados reais de SpO2:")
print(spo2_df.head())


# Renomear colunas para consistência
r_df.columns = ['R']
spo2_df.columns = ['SpO2']

# Combinar os dataframes
combined_df = pd.concat([r_df, spo2_df], axis=1)

# Verificar os dados combinados
print("\nDados combinados:")
print(combined_df)


plt.figure(figsize=(10,6))
sns.scatterplot(x='R', y='SpO2', data=combined_df, color='blue', label='Dados Reais')
plt.title('SpO2 Real vs. R Calculado')
plt.xlabel('R (AC/DC Red / AC/DC Blue)')
plt.ylabel('SpO2 Real (%)')
plt.legend()
plt.show()

# Preparar os dados para regressão
X = combined_df[['R']].values  # Feature
y = combined_df['SpO2'].values # Target

# Criar e treinar o modelo de regressão linear
linear_model = LinearRegression()
linear_model.fit(X, y)

# Fazer previsões
y_pred_linear = linear_model.predict(X)

# Avaliar o modelo
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

print("Regressão Linear Simples:")
print(f"Coeficiente Angular (B): {linear_model.coef_[0]:.4f}")
print(f"Intercepto (A): {linear_model.intercept_:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"R²: {r2_linear:.4f}\n")

# Definir o grau do polinômio
degree = 2

# Transformar as features para incluir termos polinomiais
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Criar e treinar o modelo de regressão polinomial
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Fazer previsões
y_pred_poly = poly_model.predict(X_poly)

# Avaliar o modelo
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

print(f"Regressão Polinomial de Grau {degree}:")
print(f"Coeficientes: {poly_model.coef_}")
print(f"Intercepto: {poly_model.intercept_}")
print(f"MSE: {mse_poly:.4f}")
print(f"R²: {r2_poly:.4f}\n")


plt.figure(figsize=(10,6))
sns.scatterplot(x='R', y='SpO2', data=combined_df, color='blue', label='Dados Reais')

# Plotar a linha de regressão linear
plt.plot(combined_df['R'], y_pred_linear, color='red', label='Regressão Linear')
plt.title('SpO2 Real vs. R Calculado com Regressão Linear')
plt.xlabel('R (AC/DC Red / AC/DC Blue)')
plt.ylabel('SpO2 Real (%)')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='R', y='SpO2', data=combined_df, color='blue', label='Dados Reais')

# Ordenar os valores de R para uma linha suave
sorted_df = combined_df.sort_values(by='R')
X_sorted = sorted_df[['R']].values
X_sorted_poly = poly_features.transform(X_sorted)
y_sorted_poly = poly_model.predict(X_sorted_poly)

# Plotar a curva de regressão polinomial
plt.plot(sorted_df['R'], y_sorted_poly, color='green', label=f'Regressão Polinomial Grau {degree}')
plt.title(f'SpO2 Real vs. R Calculado com Regressão Polinomial Grau {degree}')
plt.xlabel('R (AC/DC Red / AC/DC Blue)')
plt.ylabel('SpO2 Real (%)')
plt.legend()
plt.show()


# Salvar os coeficientes da regressão linear
with open('calibration_coefficients_linear.txt', 'w') as f:
    f.write(f"Intercepto (A): {linear_model.intercept_}\n")
    f.write(f"Coeficiente Angular (B): {linear_model.coef_[0]}\n")

print("Coeficientes da regressão linear salvos em 'calibration_coefficients_linear.txt'\n")


# Salvar os coeficientes da regressão polinomial
with open('calibration_coefficients_polynomial.txt', 'w') as f:
    f.write(f"Intercepto: {poly_model.intercept_}\n")
    for i in range(1, degree + 1):
        f.write(f"Coeficiente para R^{i}: {poly_model.coef_[i]}\n")

print(f"Coeficientes da regressão polinomial de grau {degree} salvos em 'calibration_coefficients_polynomial.txt'\n")
