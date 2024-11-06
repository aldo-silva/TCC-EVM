// ImageFFT.h

#ifndef IMAGE_FFT_H
#define IMAGE_FFT_H

#include <opencv2/opencv.hpp>
#include <complex>
#include <valarray>

typedef std::complex<double> Complex;
typedef std::valarray<Complex> ComplexArray;

class ImageFFT {
public:
    // Funções estáticas para cálculo da FFT e IFFT
    static void fft(ComplexArray& data);
    static void ifft(ComplexArray& data);

    // Funções para processar um canal de imagem e aplicar FFT ou IFFT
    static ComplexArray calculateFFT(const cv::Mat& channel);
    static void calculateIFFT(ComplexArray& data, cv::Mat& channel);
};

#endif // IMAGE_FFT_H
