// SignalProcessor.hpp
#ifndef SIGNALPROCESSOR_HPP
#define SIGNALPROCESSOR_HPP

#include <deque>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace my {

class SignalProcessor {
public:
    SignalProcessor(size_t bufferSize = 300);
    ~SignalProcessor();

    // Já existente
    void addFrameData(const std::vector<cv::Mat>& rgb_channels);

    // Gets
    const std::deque<double>& getRedChannelMeans() const;
    const std::deque<double>& getGreenChannelMeans() const;
    const std::deque<double>& getBlueChannelMeans() const;

    double computeHeartRate(double fps);
    double computeSpO2();
    void reset();

private:
    size_t maxBufferSize;
    std::deque<double> redChannelMeans;
    std::deque<double> greenChannelMeans;
    std::deque<double> blueChannelMeans;

    std::deque<double> redACHistory;
    std::deque<double> redDCHistory;
    std::deque<double> blueACHistory;
    std::deque<double> blueDCHistory;
    std::deque<double> RHistory;    // Armazena o valor de R
    std::deque<double> spo2History; // Armazena o valor de SpO2

    // Métodos auxiliares
    void detrend(std::deque<double>& signal);
    void applyHammingWindow(std::deque<double>& signal);
    void normalizeSignal(std::deque<double>& signal);
    double computeDominantFrequency(const std::deque<double>& inputSignal, double fps);
    void saveSignal(const std::deque<double>& signal, const std::string& filename);

    // Interpolação
    std::deque<double> linearInterpolation(const std::deque<double>& signal,
                                           double originalFps, double targetFps);


    void saveIntermediateParameters(const std::string& filename);

};

} // namespace my

#endif // SIGNALPROCESSOR_HPP
