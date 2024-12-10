// SignalProcessor.hpp
#ifndef SIGNALPROCESSOR_HPP
#define SIGNALPROCESSOR_HPP

#include <deque>
#include <vector>
#include <opencv2/opencv.hpp>

namespace my {

class SignalProcessor {
public:
    SignalProcessor(size_t bufferSize);
    ~SignalProcessor();

    void addFrameData(const std::vector<cv::Mat>& rgb_channels);

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

    void detrend(std::deque<double>& signal);
    void applyHammingWindow(std::deque<double>& signal);
    void normalizeSignal(std::deque<double>& signal);
    double computeDominantFrequency(const std::deque<double>& inputSignal, double fps);

    // Function to save signals to files
    void saveSignal(const std::deque<double>& signal, const std::string& filename);
};

} // namespace my

#endif // SIGNALPROCESSOR_HPP
