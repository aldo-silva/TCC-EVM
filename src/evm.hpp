#ifndef EVM_HPP
#define EVM_HPP

#include <opencv2/opencv.hpp>

namespace my {

class evm {
public:
    evm() = default;
    ~evm() = default;

    // Aplica FFT a uma imagem de único canal (CV_32FC1) e retorna CV_32FC2
    cv::Mat applyFFT(const cv::Mat& input) const;

    // Aplica IFFT a uma imagem de dois canais (CV_32FC2) e retorna CV_32FC1
    cv::Mat applyIFFT(const cv::Mat& input) const;

    // Filtro passa-banda simples (ou dual low pass de acordo com a lógica)
    cv::Mat applyDualLowPassFilter(const cv::Mat& dft_img, float lowCutoffFreq, float highCutoffFreq, float fps) const;

    // Amplifica a imagem
    cv::Mat ampImg(const cv::Mat& image, float alpha) const;

    // Pirâmide Gaussiana
    cv::Mat pyr_up(const cv::Mat& img) const;
    cv::Mat pyr_down(const cv::Mat& img) const;

    // Processa um único canal
    cv::Mat processChannel(const cv::Mat& channel, float lowFreq, float highFreq, float fps, float alpha) ;
};

// Função para trocar quadrantes do espectro
void fftShift(const cv::Mat& input, cv::Mat& output);

} // namespace my

#endif // EVM_HPP
