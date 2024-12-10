import matplotlib.pyplot as plt

def plot_signal(filename, title):
    indices = []
    values = []
    with open(filename, 'r') as file:
        for line in file:
            index, value = line.strip().split(',')
            indices.append(float(index))
            values.append(float(value))
    plt.figure()
    plt.plot(indices, values)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Plot signals at different stages
plot_signal('data/raw_signal.csv', 'Raw Signal')
plot_signal('data/detrended_signal.csv', 'Signal after Detrending')
plot_signal('data/windowed_signal.csv', 'Signal after Hamming Window')
plot_signal('data/normalized_signal.csv', 'Normalized Signal')

# Plot magnitude spectrum
def plot_spectrum(filename, title):
    freqs = []
    mags = []
    with open(filename, 'r') as file:
        for line in file:
            freq, mag = line.strip().split(',')
            freqs.append(float(freq))
            mags.append(float(mag))
    plt.figure()
    plt.plot(freqs, mags)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

plot_spectrum('data/magnitude_spectrum.csv', 'Magnitude Spectrum')
