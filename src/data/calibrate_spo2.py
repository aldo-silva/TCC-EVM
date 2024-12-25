import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Configurações de visualização
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data(r_csv_path, spo2_csv_path):
    """
    Carrega os dados de R e SpO2 de arquivos CSV.
    """
    try:
        r_df = pd.read_csv(r_csv_path)
        spo2_df = pd.read_csv(spo2_csv_path)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return None, None
    
    # Verificar se os arquivos têm a mesma quantidade de linhas
    if len(r_df) != len(spo2_df):
        print("Erro: Os arquivos CSV têm quantidades diferentes de linhas.")
        return None, None
    
    # Renomear colunas para garantir consistência
    r_df.columns = ['R']
    spo2_df.columns = ['SpO2']
    
    # Combinar os dataframes
    combined_df = pd.concat([r_df, spo2_df], axis=1)
    
    # Remover possíveis linhas com valores faltantes
    combined_df.dropna(inplace=True)
    
    return combined_df

def plot_data(df):
    """
    Plota os dados de SpO2 vs R.
    """
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='R', y='SpO2', data=df, color='blue', label='Dados Reais')
    plt.title('SpO2 Real vs. R Calculado')
    plt.xlabel('R (AC/DC Red / AC/DC Blue)')
    plt.ylabel('SpO2 Real (%)')
    plt.legend()
    plt.show()

def perform_linear_regression(X, y):
    """
    Realiza regressão linear simples.
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Avaliação do modelo
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("Regressão Linear Simples:")
    print(f"Coeficiente Angular (B): {model.coef_[0]:.4f}")
    print(f"Intercepto (A): {model.intercept_:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}\n")
    
    return model

def perform_polynomial_regression(X, y, degree=2):
    """
    Realiza regressão polinomial de grau especificado.
    """
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # Avaliação do modelo
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Regressão Polinomial de Grau {degree}:")
    print(f"Coeficientes: {model.coef_}")
    print(f"Intercepto: {model.intercept_}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}\n")
    
    return model, poly

def plot_regression(df, model, poly=None, degree=2):
    """
    Plota os dados e a curva de regressão.
    """
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='R', y='SpO2', data=df, color='blue', label='Dados Reais')
    
    # Ordenar os valores de R para uma linha suave
    sorted_df = df.sort_values(by='R')
    X_sorted = sorted_df[['R']].values
    y_sorted = sorted_df['SpO2'].values
    
    if poly:
        X_plot = poly.transform(X_sorted)
        y_plot = model.predict(X_plot)
        label = f'Regressão Polinomial Grau {degree}'
    else:
        y_plot = model.predict(X_sorted)
        label = 'Regressão Linear'
    
    plt.plot(sorted_df['R'], y_plot, color='red', label=label)
    plt.title('SpO2 Real vs. R Calculado com Regressão')
    plt.xlabel('R (AC/DC Red / AC/DC Blue)')
    plt.ylabel('SpO2 Real (%)')
    plt.legend()
    plt.show()

def save_coefficients_linear(model, filename='calibration_coefficients_linear.txt'):
    """
    Salva os coeficientes da regressão linear em um arquivo de texto.
    """
    with open(filename, 'w') as f:
        f.write(f"Intercepto (A): {model.intercept_}\n")
        f.write(f"Coeficiente Angular (B): {model.coef_[0]}\n")
    print(f"Coeficientes da regressão linear salvos em {filename}\n")

def save_coefficients_polynomial(model, poly, degree, filename='calibration_coefficients_polynomial.txt'):
    """
    Salva os coeficientes da regressão polinomial em um arquivo de texto.
    """
    with open(filename, 'w') as f:
        f.write(f"Intercepto: {model.intercept_}\n")
        for i in range(1, degree+1):
            f.write(f"Coeficiente para R^{i}: {model.coef_[i]}\n")
    print(f"Coeficientes da regressão polinomial de grau {degree} salvos em {filename}\n")

def main():
    # Caminhos para os arquivos CSV
    r_csv_path = 'R_values.csv'        # Atualize com o caminho correto
    spo2_csv_path = 'SpO2_real.csv'    # Atualize com o caminho correto
    
    # Carregar os dados
    df = load_data(r_csv_path, spo2_csv_path)
    if df is None or df.empty:
        print("Erro ao carregar os dados. Verifique os arquivos CSV.")
        return
    
    print("Dados carregados com sucesso.\n")
    
    # Visualizar os dados
    plot_data(df)
    
    # Preparar os dados para regressão
    X = df[['R']].values  # Feature
    y = df['SpO2'].values # Target
    
    # Regressão Linear Simples
    linear_model = perform_linear_regression(X, y)
    
    # Regressão Polinomial de Grau 2 (se necessário)
    degree = 2
    poly_model, poly_features = perform_polynomial_regression(X, y, degree=degree)
    
    # Plotar Regressão Linear
    plot_regression(df, linear_model)
    
    # Plotar Regressão Polinomial
    plot_regression(df, poly_model, poly=poly_features, degree=degree)
    
    # Salvar os coeficientes
    save_coefficients_linear(linear_model, filename='calibration_coefficients_linear.txt')
    save_coefficients_polynomial(poly_model, poly_features, degree=degree, filename='calibration_coefficients_polynomial.txt')
    
    # Escolha qual modelo usar baseado na performance
    # Por exemplo, se a regressão linear já tem um bom R², você pode optar por ela.
    
    # Exemplo de como usar os coeficientes para calcular SpO2
    # Com regressão linear: SpO2 = A + B * R
    A = linear_model.intercept_
    B = linear_model.coef_[0]
    print(f"Equação de Regressão Linear: SpO2 = {A:.4f} + {B:.4f} * R")
    
    # Ou com regressão polinomial de grau 2: SpO2 = a0 + a1 * R + a2 * R^2
    a0 = poly_model.intercept_
    a1 = poly_model.coef_[1]
    a2 = poly_model.coef_[2]
    print(f"Equação de Regressão Polinomial Grau {degree}: SpO2 = {a0:.4f} + {a1:.4f} * R + {a2:.4f} * R^2")

if __name__ == "__main__":
    main()
