// SignalProcessor.cpp
#include "SignalProcessor.hpp"
#include <numeric>
#include <algorithm>
#include <fftw3.h>
#include <cmath>    // For M_PI
#include <fstream>  // For file operations
#include <iostream> // For std::cerr

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

    // Save the channel means after updating
    saveSignal(redChannelMeans, "/home/aldo/data/redChannelMeans.csv");
    saveSignal(greenChannelMeans, "/home/aldo/datagreenChannelMeans.csv");
    saveSignal(blueChannelMeans, "/home/aldo/data/blueChannelMeans.csv");
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
    int N = signal.size();
    if (N < 2) return;

    // Compute linear fit (least squares)
    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumXY = 0.0;
    for (int i = 0; i < N; ++i) {
        sumX += i;
        sumY += signal[i];
        sumX2 += i * i;
        sumXY += i * signal[i];
    }
    double denom = N * sumX2 - sumX * sumX;
    if (denom == 0) return; // Prevent division by zero

    double a = (N * sumXY - sumX * sumY) / denom;
    double b = (sumY * sumX2 - sumX * sumXY) / denom;

    // Remove the trend
    for (int i = 0; i < N; ++i) {
        double trend = a * i + b;
        signal[i] -= trend;
    }
}

void SignalProcessor::applyHammingWindow(std::deque<double>& signal) {
    int N = signal.size();
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

double SignalProcessor::computeDominantFrequency(const std::deque<double>& inputSignal, double fps) {
    if (inputSignal.size() < 2) {
        return -1.0; // Insufficient data
    }

    // Copy the input signal
    std::deque<double> signal = inputSignal;

    // Save the raw signal
    saveSignal(signal, "/home/aldo/data/raw_signal.csv");

    // Step 1: Detrend the signal
    detrend(signal);
    saveSignal(signal, "/home/aldo/data/detrended_signal.csv");

    // Step 2: Apply Hamming window
    applyHammingWindow(signal);
    saveSignal(signal, "/home/aldo/data/windowed_signal.csv");

    // Step 3: Normalize the signal
    normalizeSignal(signal);
    saveSignal(signal, "/home/aldo/data/normalized_signal.csv");

    // Prepare data for FFT
    int N = signal.size();
    double* in = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    std::copy(signal.begin(), signal.end(), in);

    // Execute FFT
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(p);

    // Compute magnitude spectrum
    std::vector<double> magnitudes(N / 2 + 1);
    for (int i = 0; i < N / 2 + 1; ++i) {
        magnitudes[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }

    // Save the magnitude spectrum
    double freqResolution = fps / N;
    std::ofstream magFile("/home/aldo/data/magnitude_spectrum.csv");
    if (magFile.is_open()) {
        for (size_t i = 0; i < magnitudes.size(); ++i) {
            double freq = i * freqResolution;
            magFile << freq << "," << magnitudes[i] << std::endl;
        }
        magFile.close();
    } else {
        std::cerr << "Unable to open file magnitude_spectrum.csv for writing." << std::endl;
    }

    // Find the dominant frequency within the expected heart rate range (e.g., 0.8 - 3.0 Hz)
    int minIndex = static_cast<int>(0.8 / freqResolution);
    int maxIndex = static_cast<int>(3.0 / freqResolution);

    double max_magnitude = 0.0;
    int max_index = minIndex;

    for (int i = minIndex; i <= maxIndex && i < magnitudes.size(); ++i) {
        if (magnitudes[i] > max_magnitude) {
            max_magnitude = magnitudes[i];
            max_index = i;
        }
    }

    double minFrequency = 0.8;
    double maxFrequency = 3.0;

    std::ofstream cutoffFile("/home/aldo/data/cutoff_frequencies.txt");
    if (cutoffFile.is_open()) {
        cutoffFile << minFrequency << "," << maxFrequency;
        cutoffFile.close();
    } else {
        std::cerr << "Unable to open file cutoff_frequencies.txt for writing." << std::endl;
    }

    // Convert index to frequency
    double frequency = max_index * freqResolution; // in Hz

    std::ofstream freqFile("/home/aldo/data/dominant_frequency.txt");
    if (freqFile.is_open()) {
        freqFile << frequency;
        freqFile.close();
    } else {
        std::cerr << "Unable to open file dominant_frequency.txt for writing." << std::endl;
    }

    // Clean up
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return frequency;
}

double SignalProcessor::computeHeartRate(double fps) {
    double frequency = computeDominantFrequency(greenChannelMeans, fps);
    if (frequency <= 0.0) {
        return -1.0; // Invalid frequency
    }
    double heartRate = frequency * 60.0; // Convert Hz to bpm
    return heartRate;
}

double SignalProcessor::computeSpO2() {
    if (redChannelMeans.size() < 2 || blueChannelMeans.size() < 2) {
        return -1.0;
    }

    // Calculate AC and DC components for red and blue channels
    double redMean = std::accumulate(redChannelMeans.begin(), redChannelMeans.end(), 0.0) / redChannelMeans.size();
    double blueMean = std::accumulate(blueChannelMeans.begin(), blueChannelMeans.end(), 0.0) / blueChannelMeans.size();

    // Calculate AC components (standard deviation)
    double redSumSq = 0.0;
    for (double val : redChannelMeans) {
        redSumSq += (val - redMean) * (val - redMean);
    }
    double redAC = sqrt(redSumSq / redChannelMeans.size());

    double blueSumSq = 0.0;
    for (double val : blueChannelMeans) {
        blueSumSq += (val - blueMean) * (val - blueMean);
    }
    double blueAC = sqrt(blueSumSq / blueChannelMeans.size());

    double R = (redAC / redMean) / (blueAC / blueMean);

    // Use empirical formula (simplified example)
    double spo2 = 110 - 25 * R; // This formula is a simplification and may not be accurate

    // Clamp the value between 0 and 100
    spo2 = std::max(0.0, std::min(100.0, spo2));

    return spo2;
}

void SignalProcessor::reset() {
    redChannelMeans.clear();
    greenChannelMeans.clear();
    blueChannelMeans.clear();
}

} // namespace my
