import pandas as pd
import matplotlib.pyplot as plt

magnitude_spectrum_path = "/home/aldo/data/magnitude_spectrum.csv"
dominant_frequency_path = "/home/aldo/data/dominant_frequency.txt"
cutoff_frequencies_path = "/home/aldo/data/cutoff_frequencies.txt"

spectrum = pd.read_csv(magnitude_spectrum_path, header=None, names=['Frequency', 'Magnitude'])

with open(dominant_frequency_path, 'r') as f:
    dominant_freq = float(f.read().strip())

with open(cutoff_frequencies_path, 'r') as f:
    cutoffs = f.read().strip().split(',')
    min_freq = float(cutoffs[0])
    max_freq = float(cutoffs[1])

plt.figure(figsize=(12, 6))
plt.plot(spectrum['Frequency'], spectrum['Magnitude'], label='Espectro de Magnitude')

plt.axvline(x=min_freq, color='red', linestyle='--', label=f'Frequência Mínima ({min_freq} Hz)')
plt.axvline(x=max_freq, color='green', linestyle='--', label=f'Frequência Máxima ({max_freq} Hz)')

plt.axvline(x=dominant_freq, color='purple', linestyle='-', linewidth=2, label=f'Frequência Dominante ({dominant_freq:.2f} Hz)')

plt.title('Espectro de Magnitude com Frequências de Corte e Dominante')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.savefig('/home/aldo/data/magnitude_spectrum_plot.png')

plt.show()