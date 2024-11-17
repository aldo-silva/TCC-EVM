#ifndef SIGNAL_PROCESSOR_HPP
#define SIGNAL_PROCESSOR_HPP

#include <vector>
#include <deque>
#include <opencv2/opencv.hpp>

namespace my {

class SignalProcessor {
public:
    SignalProcessor(size_t bufferSize = 300); // Default buffer size to store 10 seconds at 30 fps
    ~SignalProcessor();

    // Add a new frame's RGB channels
    void addFrameData(const std::vector<cv::Mat>& rgb_channels);

    // Get the mean values over time for each channel
    const std::deque<double>& getRedChannelMeans() const;
    const std::deque<double>& getGreenChannelMeans() const;
    const std::deque<double>& getBlueChannelMeans() const;

    // Get the standard deviation values over time for each channel
    const std::deque<double>& getRedChannelStdDevs() const;
    const std::deque<double>& getGreenChannelStdDevs() const;
    const std::deque<double>& getBlueChannelStdDevs() const;

    // Compute heart rate using the green channel
    double computeHeartRate(double fps);

    // Compute SpO₂ using red and blue channels
    double computeSpO2();

    // Reset the stored data
    void reset();

private:
    // Circular buffers to store mean and std dev over time
    std::deque<double> redChannelMeans;
    std::deque<double> greenChannelMeans;
    std::deque<double> blueChannelMeans;

    std::deque<double> redChannelStdDevs;
    std::deque<double> greenChannelStdDevs;
    std::deque<double> blueChannelStdDevs;

    // Maximum buffer size
    size_t maxBufferSize;

    // Helper function to compute frequency
    double computeDominantFrequency(const std::deque<double>& signal, double fps);

};

} // namespace my

#endif // SIGNAL_PROCESSOR_HPP
