# spo2_analysis.ipynb (Exemplo de notebook em Python)

import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar dados do CSV
# Ajuste o caminho conforme necessário
filename = "/home/aldo/data/spo2_intermediate_params.csv"

df = pd.read_csv(filename)

# df.columns deve ser algo como:
# ["Index", "RedAC", "RedDC", "BlueAC", "BlueDC", "R", "SpO2"]

# 2. Plotar as curvas

# Ajuste o tamanho das figuras
plt.figure(figsize=(12, 8))

# 2.1 Plot do AC vermelho e azul
plt.subplot(2, 2, 1)
plt.plot(df["Index"], df["RedAC"], label='RedAC', color='red')
plt.plot(df["Index"], df["BlueAC"], label='BlueAC', color='blue')
plt.title("AC Components (std dev)")
plt.xlabel("Index")
plt.ylabel("AC Value")
plt.legend()

# 2.2 Plot do DC vermelho e azul
plt.subplot(2, 2, 2)
plt.plot(df["Index"], df["RedDC"], label='RedDC', color='red')
plt.plot(df["Index"], df["BlueDC"], label='BlueDC', color='blue')
plt.title("DC Components (mean)")
plt.xlabel("Index")
plt.ylabel("DC Value")
plt.legend()

# 2.3 Plot do R
plt.subplot(2, 2, 3)
plt.plot(df["Index"], df["R"], label='R', color='green')
plt.title("R = (ACr/DCr) / (ACb/DCb)")
plt.xlabel("Index")
plt.ylabel("R")
plt.legend()

# 2.4 Plot do SpO2
plt.subplot(2, 2, 4)
plt.plot(df["Index"], df["SpO2"], label='SpO2', color='purple')
plt.title("Estimated SpO2 (%)")
plt.xlabel("Index")
plt.ylabel("SpO2 (%)")
plt.ylim(0, 110)   # Ajuste para visualizar melhor
plt.legend()

plt.tight_layout()
plt.show()
