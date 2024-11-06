// ImageFFT.cpp

#include "ImageFFT.hpp"
#include <cmath>

const double PI = 3.14159265358979323846;

// Implementação da FFT
void ImageFFT::fft(ComplexArray& x) {
    size_t N = x.size();
    if (N <= 1) return;

    ComplexArray even = x[std::slice(0, N / 2, 2)];
    ComplexArray odd = x[std::slice(1, N / 2, 2)];

    fft(even);
    fft(odd);

    for (size_t i = 0; i < N / 2; ++i) {
        Complex t = std::polar(1.0, -2 * PI * i / N) * odd[i];
        x[i] = even[i] + t;
        x[i + N / 2] = even[i] - t;
    }
}

// Implementação da IFFT
void ImageFFT::ifft(ComplexArray& x) {
    x = x.apply(std::conj);
    fft(x);
    x = x.apply(std::conj);
    x /= x.size();
}

// Converter um canal de imagem para um ComplexArray e calcular FFT
ComplexArray ImageFFT::calculateFFT(const cv::Mat& channel) {
    Complex* signal = new Complex[channel.cols * channel.rows];
    for (int i = 0; i < channel.rows; i++) {
        for (int j = 0; j < channel.cols; j++) {
            signal[i * channel.cols + j] = Complex(channel.at<uchar>(i, j), 0);
        }
    }

    ComplexArray data(signal, channel.cols * channel.rows);
    fft(data);
    delete[] signal;
    return data;
}

// Converter um ComplexArray de volta para um canal de imagem usando IFFT
void ImageFFT::calculateIFFT(ComplexArray& data, cv::Mat& channel) {
    ifft(data);
    for (int i = 0; i < channel.rows; i++) {
        for (int j = 0; j < channel.cols; j++) {
            channel.at<uchar>(i, j) = (uchar)std::real(data[i * channel.cols + j]);
        }
    }
}

