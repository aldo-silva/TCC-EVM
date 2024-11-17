#include "SignalProcessor.hpp"
#include <numeric>
#include <algorithm>
#include <fftw3.h>

namespace my {

SignalProcessor::SignalProcessor(size_t bufferSize)
    : maxBufferSize(bufferSize) {
    // Constructor
}

SignalProcessor::~SignalProcessor() {
    // Destructor
}

void SignalProcessor::addFrameData(const std::vector<cv::Mat>& rgb_channels) {
    // Compute mean and std dev for each channel
    for (int i = 0; i < 3; ++i) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(rgb_channels[i], mean, stddev);

        switch (i) {
            case 0: // Blue channel
                blueChannelMeans.push_back(mean[0]);
                blueChannelStdDevs.push_back(stddev[0]);
                if (blueChannelMeans.size() > maxBufferSize) {
                    blueChannelMeans.pop_front();
                    blueChannelStdDevs.pop_front();
                }
                break;
            case 1: // Green channel
                greenChannelMeans.push_back(mean[0]);
                greenChannelStdDevs.push_back(stddev[0]);
                if (greenChannelMeans.size() > maxBufferSize) {
                    greenChannelMeans.pop_front();
                    greenChannelStdDevs.pop_front();
                }
                break;
            case 2: // Red channel
                redChannelMeans.push_back(mean[0]);
                redChannelStdDevs.push_back(stddev[0]);
                if (redChannelMeans.size() > maxBufferSize) {
                    redChannelMeans.pop_front();
                    redChannelStdDevs.pop_front();
                }
                break;
        }
    }
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

const std::deque<double>& SignalProcessor::getRedChannelStdDevs() const {
    return redChannelStdDevs;
}

const std::deque<double>& SignalProcessor::getGreenChannelStdDevs() const {
    return greenChannelStdDevs;
}

const std::deque<double>& SignalProcessor::getBlueChannelStdDevs() const {
    return blueChannelStdDevs;
}

double SignalProcessor::computeDominantFrequency(const std::deque<double>& signal, double fps) {
    if (signal.size() < 2) {
        return -1.0; // Not enough data
    }

    int N = signal.size();
    double* in = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2 + 1));

    // Prepare the input data
    std::copy(signal.begin(), signal.end(), in);

    // Perform FFT
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(p);

    // Compute magnitude spectrum
    std::vector<double> magnitudes(N/2 + 1);
    for (int i = 0; i < N/2 + 1; ++i) {
        magnitudes[i] = sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
    }

    // Find the peak frequency within expected heart rate range (e.g., 0.8 - 3.0 Hz)
    double freqResolution = fps / N;
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

    // Convert index to frequency
    double frequency = max_index * freqResolution; // in Hz

    // Cleanup
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

    double redAC = std::sqrt(std::accumulate(redChannelStdDevs.begin(), redChannelStdDevs.end(), 0.0,
        [](double sum, double val){ return sum + val * val; }) / redChannelStdDevs.size());
    double blueAC = std::sqrt(std::accumulate(blueChannelStdDevs.begin(), blueChannelStdDevs.end(), 0.0,
        [](double sum, double val){ return sum + val * val; }) / blueChannelStdDevs.size());

    double R = (redAC / redMean) / (blueAC / blueMean);

    // Use empirical formula (placeholder)
    double spo2 = 110 - 25 * R; // This formula is a simplification and may not be accurate

    // Clamp the value between 0 and 100
    spo2 = std::max(0.0, std::min(100.0, spo2));

    return spo2;
}

void SignalProcessor::reset() {
    redChannelMeans.clear();
    greenChannelMeans.clear();
    blueChannelMeans.clear();

    redChannelStdDevs.clear();
    greenChannelStdDevs.clear();
    blueChannelStdDevs.clear();
}

} // namespace my
