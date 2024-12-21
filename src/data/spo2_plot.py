# spo2_analysis.ipynb

import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar dados do CSV
# Ajuste o caminho conforme necessário
filename = "/home/aldo/data/spo2_intermediate_params.csv"

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Arquivo {filename} não encontrado. Certifique-se de que o caminho está correto.")
    raise

# Verificar as primeiras linhas para garantir que os dados foram carregados corretamente
print(df.head())

# 2. Plotar as curvas

# Ajustar o tamanho das figuras
plt.figure(figsize=(15, 12))

# 2.1 Plot do AC vermelho e azul
plt.subplot(3, 2, 1)
plt.plot(df["Index"], df["RedAC"], label='RedAC', color='red')
plt.plot(df["Index"], df["BlueAC"], label='BlueAC', color='blue')
plt.title("AC Components (Standard Deviation)")
plt.xlabel("Frame Index")
plt.ylabel("AC Value")
plt.legend()
plt.grid(True)

# 2.2 Plot do DC vermelho e azul
plt.subplot(3, 2, 2)
plt.plot(df["Index"], df["RedDC"], label='RedDC', color='orange')
plt.plot(df["Index"], df["BlueDC"], label='BlueDC', color='cyan')
plt.title("DC Components (Mean)")
plt.xlabel("Frame Index")
plt.ylabel("DC Value")
plt.legend()
plt.grid(True)

# 2.3 Plot do R
plt.subplot(3, 2, 3)
plt.plot(df["Index"], df["R"], label='R', color='green')
plt.title("R = (ACr/DCr) / (ACb/DCb)")
plt.xlabel("Frame Index")
plt.ylabel("R Value")
plt.legend()
plt.grid(True)

# 2.4 Plot do SpO2
plt.subplot(3, 2, 4)
plt.plot(df["Index"], df["SpO2"], label='SpO2', color='purple')
plt.title("Estimated SpO2 (%)")
plt.xlabel("Frame Index")
plt.ylabel("SpO2 (%)")
plt.ylim(0, 100)   # Ajuste para visualizar melhor
plt.legend()
plt.grid(True)

# 2.5 Plot da Magnitude Spectrum (opcional)
# Se desejar, você pode também plotar a magnitude do espectro
# Verifique se você salvou o 'magnitude_spectrum.csv'

mag_filename = "/home/aldo/data/magnitude_spectrum.csv"
try:
    mag_df = pd.read_csv(mag_filename, header=None, names=['Frequency', 'Magnitude'])
    plt.subplot(3, 2, 5)
    plt.plot(mag_df["Frequency"], mag_df["Magnitude"], color='magenta')
    plt.title("Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 5)  # Ajuste conforme necessário
    plt.grid(True)
except FileNotFoundError:
    print(f"Arquivo {mag_filename} não encontrado. Este gráfico será omitido.")

# 2.6 Plot das Frequências de Corte (opcional)
# Se desejar, você pode também plotar as frequências de corte
cutoff_filename = "/home/aldo/data/cutoff_frequencies.txt"
try:
    with open(cutoff_filename, 'r') as f:
        minFreq, maxFreq = map(float, f.read().strip().split(','))
    plt.subplot(3, 2, 6)
    plt.axvline(x=minFreq, color='grey', linestyle='--', label='Min Cutoff')
    plt.axvline(x=maxFreq, color='black', linestyle='--', label='Max Cutoff')
    plt.title("Cutoff Frequencies")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
except FileNotFoundError:
    print(f"Arquivo {cutoff_filename} não encontrado. Este gráfico será omitido.")

plt.tight_layout()
plt.show()
