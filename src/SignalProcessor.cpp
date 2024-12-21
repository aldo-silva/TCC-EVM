// SignalProcessor.cpp
#include "SignalProcessor.hpp"
#include <numeric>
#include <algorithm>
#include <fftw3.h>
#include <cmath>    
#include <fstream>  
#include <iostream> 

namespace my {

SignalProcessor::SignalProcessor(size_t bufferSize)
    : maxBufferSize(bufferSize) {
    // Constructor
}

SignalProcessor::~SignalProcessor() {
    // Destructor
}

void SignalProcessor::addFrameData(const std::vector<cv::Mat>& rgb_channels) {
    // Compute the mean for each channel
    for (int i = 0; i < 3; ++i) {
        cv::Scalar mean;
        mean = cv::mean(rgb_channels[i]);

        switch (i) {
            case 0: // Blue channel
                blueChannelMeans.push_back(mean[0]);
                if (blueChannelMeans.size() > maxBufferSize) {
                    blueChannelMeans.pop_front();
                }
                break;
            case 1: // Green channel
                greenChannelMeans.push_back(mean[0]);
                if (greenChannelMeans.size() > maxBufferSize) {
                    greenChannelMeans.pop_front();
                }
                break;
            case 2: // Red channel
                redChannelMeans.push_back(mean[0]);
                if (redChannelMeans.size() > maxBufferSize) {
                    redChannelMeans.pop_front();
                }
                break;
        }
    }

    // Salva os meios de cada canal para debug
    saveSignal(redChannelMeans,   "/home/aldo/data/redChannelMeans.csv");
    saveSignal(greenChannelMeans, "/home/aldo/data/greenChannelMeans.csv");
    saveSignal(blueChannelMeans,  "/home/aldo/data/blueChannelMeans.csv");
}

const std::deque<double>& SignalProcessor::getRedChannelMeans() const {
    return redChannelMeans;
}

const std::deque<double>& SignalProcessor::getGreenChannelMeans() const {
    return greenChannelMeans;
}

const std::deque<double>& SignalProcessor::getBlueChannelMeans() const {
    return blueChannelMeans;
}

void SignalProcessor::detrend(std::deque<double>& signal) {
    int N = (int)signal.size();
    if (N < 2) return;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumXY = 0.0;
    for (int i = 0; i < N; ++i) {
        sumX  += i;
        sumY  += signal[i];
        sumX2 += (double)i * i;
        sumXY += i * signal[i];
    }
    double denom = N * sumX2 - sumX * sumX;
    if (denom == 0) return; 

    double a = (N * sumXY - sumX * sumY) / denom;
    double b = (sumY * sumX2 - sumX * sumXY) / denom;

    // Remove the trend
    for (int i = 0; i < N; ++i) {
        double trend = a * i + b;
        signal[i] -= trend;
    }
}

void SignalProcessor::applyHammingWindow(std::deque<double>& signal) {
    int N = (int)signal.size();
    for (int n = 0; n < N; ++n) {
        double w = 0.54 - 0.46 * cos(2 * M_PI * n / (N - 1));
        signal[n] *= w;
    }
}

void SignalProcessor::normalizeSignal(std::deque<double>& signal) {
    double norm = 0.0;
    for (double val : signal) {
        norm += val * val;
    }
    norm = sqrt(norm);

    if (norm > 0.0) {
        for (double& val : signal) {
            val /= norm;
        }
    }
}

void SignalProcessor::saveSignal(const std::deque<double>& signal, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < signal.size(); ++i) {
            file << i << "," << signal[i] << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
    }
}

// NOVO: Salva parâmetros intermediários
void SignalProcessor::saveIntermediateParameters(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Index,RedAC,RedDC,BlueAC,BlueDC,R,SpO2\n";
        size_t n = std::min({ redACHistory.size(), 
                              redDCHistory.size(), 
                              blueACHistory.size(),
                              blueDCHistory.size(),
                              RHistory.size(),
                              spo2History.size() });
        for (size_t i = 0; i < n; ++i) {
            file << i << ","
                 << redACHistory[i] << ","
                 << redDCHistory[i] << ","
                 << blueACHistory[i] << ","
                 << blueDCHistory[i] << ","
                 << RHistory[i] << ","
                 << spo2History[i] << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << " for writing intermediate params." << std::endl;
    }
}

// Interpolação linear
std::deque<double> SignalProcessor::linearInterpolation(const std::deque<double>& signal,
                                                        double originalFps, double targetFps) {
    if (signal.size() < 2) {
        return signal; 
    }

    double ratio = targetFps / originalFps;
    size_t newSize = static_cast<size_t>(signal.size() * ratio);
    if (newSize == 0) newSize = signal.size();

    std::deque<double> interpolatedSignal(newSize);

    for (size_t i = 0; i < newSize; ++i) {
        double t = i / targetFps;         
        double origIndex = t * originalFps;
        int index = static_cast<int>(std::floor(origIndex));
        double frac = origIndex - index;

        if (index < 0) {
            interpolatedSignal[i] = signal.front();
        } else if (index >= (int)signal.size() - 1) {
            interpolatedSignal[i] = signal.back();
        } else {
            interpolatedSignal[i] = signal[index] 
                + frac * (signal[index + 1] - signal[index]);
        }
    }
    return interpolatedSignal;
}

// FFT e frequência dominante
double SignalProcessor::computeDominantFrequency(const std::deque<double>& inputSignal, double fps) {
    if (inputSignal.size() < 2) {
        return -1.0; // Insufficient data
    }

    std::deque<double> signal = inputSignal;
    saveSignal(signal, "/home/aldo/data/raw_signal.csv");

    detrend(signal);
    saveSignal(signal, "/home/aldo/data/detrended_signal.csv");

    double targetFps = fps; 
    std::deque<double> interpolatedSignal = linearInterpolation(signal, fps, targetFps);
    saveSignal(interpolatedSignal, "/home/aldo/data/interpolated_signal.csv");

    signal = interpolatedSignal;

    applyHammingWindow(signal);
    saveSignal(signal, "/home/aldo/data/windowed_signal.csv");

    normalizeSignal(signal);
    saveSignal(signal, "/home/aldo/data/normalized_signal.csv");

    int N = (int)signal.size();
    double* in = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    std::copy(signal.begin(), signal.end(), in);

    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(p);

    std::vector<double> magnitudes(N / 2 + 1);
    for (int i = 0; i < N / 2 + 1; ++i) {
        magnitudes[i] = std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }

    double freqResolution = targetFps / N;
    {
        std::ofstream magFile("/home/aldo/data/magnitude_spectrum.csv");
        if (magFile.is_open()) {
            for (size_t i = 0; i < magnitudes.size(); ++i) {
                double freq = i * freqResolution;
                magFile << freq << "," << magnitudes[i] << std::endl;
            }
            magFile.close();
        }
    }

    double minFrequency = 0.8;
    double maxFrequency = 3.0;

    {
        std::ofstream cutoffFile("/home/aldo/data/cutoff_frequencies.txt");
        if (cutoffFile.is_open()) {
            cutoffFile << minFrequency << "," << maxFrequency;
            cutoffFile.close();
        }
    }

    int minIndex = (int)std::ceil(minFrequency / freqResolution);
    int maxIndex = (int)std::floor(maxFrequency / freqResolution);

    double max_magnitude = 0.0;
    int max_index = minIndex;

    for (int i = minIndex; i <= maxIndex && i < (int)magnitudes.size(); ++i) {
        if (magnitudes[i] > max_magnitude) {
            max_magnitude = magnitudes[i];
            max_index = i;
        }
    }

    double frequency = max_index * freqResolution; // Hz

    {
        std::ofstream freqFile("/home/aldo/data/dominant_frequency.txt");
        if (freqFile.is_open()) {
            freqFile << frequency;
            freqFile.close();
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return frequency;
}

// HeartRate usa canal verde
double SignalProcessor::computeHeartRate(double fps) {
    double frequency = computeDominantFrequency(greenChannelMeans, fps);
    if (frequency <= 0.0) {
        return -1.0;
    }
    double heartRate = frequency * 60.0; 
    return heartRate;
}

// SpO2 + salvar parâmetros intermediários
double SignalProcessor::computeSpO2() {
    if (redChannelMeans.size() < 2 || blueChannelMeans.size() < 2) {
        return -1.0;
    }

    double redMean = std::accumulate(redChannelMeans.begin(), redChannelMeans.end(), 0.0)
                     / redChannelMeans.size();
    double blueMean = std::accumulate(blueChannelMeans.begin(), blueChannelMeans.end(), 0.0)
                      / blueChannelMeans.size();

    // AC (std dev)
    double redSumSq = 0.0;
    for (double val : redChannelMeans) {
        redSumSq += (val - redMean) * (val - redMean);
    }
    double redAC = std::sqrt(redSumSq / redChannelMeans.size());

    double blueSumSq = 0.0;
    for (double val : blueChannelMeans) {
        blueSumSq += (val - blueMean) * (val - blueMean);
    }
    double blueAC = std::sqrt(blueSumSq / blueChannelMeans.size());

    // R = (redAC / redMean) / (blueAC / blueMean)
    double R = (redAC / redMean) / (blueAC / blueMean);

    // Fórmula empírica
    double spo2 = 110.0 - 25.0 * R; 
    spo2 = std::max(0.0, std::min(100.0, spo2));

    // Armazenar nos buffers
    redACHistory.push_back(redAC);
    redDCHistory.push_back(redMean);
    blueACHistory.push_back(blueAC);
    blueDCHistory.push_back(blueMean);
    RHistory.push_back(R);
    spo2History.push_back(spo2);

    // Retornamos a estimativa
    return spo2;
}

void SignalProcessor::reset() {
    redChannelMeans.clear();
    greenChannelMeans.clear();
    blueChannelMeans.clear();

    redACHistory.clear();
    redDCHistory.clear();
    blueACHistory.clear();
    blueDCHistory.clear();
    RHistory.clear();
    spo2History.clear();
}

} // namespace my
