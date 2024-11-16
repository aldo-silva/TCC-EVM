#include "evm.hpp"
#include <iostream>
#include <fftw3.h>

namespace my {

// Função para aplicar FFT usando FFTW
cv::Mat evm::applyFFT(const cv::Mat& input) const {
    std::cout << "Entrando na FFT com " << input.channels() << " canais." << std::endl;
    CV_Assert(input.type() == CV_32FC1);

    int rows = input.rows;
    int cols = input.cols;

    fftwf_complex *in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
    fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);

    // Prepara os dados para a FFT
    for (int i = 0; i < rows; ++i) {
        const float* inputRow = input.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = inputRow[j];  // Parte real
            in[idx][1] = 0.0f;         // Parte imaginária
        }
    }

    // Cria plano e executa a FFT
    fftwf_plan p = fftwf_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // Converte o resultado para cv::Mat de 2 canais (parte real e imaginária)
    cv::Mat dft_img(rows, cols, CV_32FC2);
    for (int i = 0; i < rows; ++i) {
        cv::Vec2f* dftRow = dft_img.ptr<cv::Vec2f>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            dftRow[j][0] = out[idx][0]; // Parte real
            dftRow[j][1] = out[idx][1]; // Parte imaginária
        }
    }

    // Centralização dos dados da FFT (shifting quadrants)
    cv::Mat shifted_dft;
    cv::fftShift(dft_img, shifted_dft);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    std::cout << "Saindo da FFT com " << shifted_dft.channels() << " canais." << std::endl;
    return shifted_dft;
}

// Função para aplicar IFFT usando FFTW
cv::Mat evm::applyIFFT(const cv::Mat& input) const {
    std::cout << "Entrando na IFFT com " << input.channels() << " canais." << std::endl;
    CV_Assert(input.type() == CV_32FC2);

    int rows = input.rows;
    int cols = input.cols;

    // Descentralizar os dados antes da IFFT
    cv::Mat input_descentralized;
    cv::fftShift(input, input_descentralized);

    fftwf_complex *in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
    fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);

    // Prepara os dados para a IFFT
    for (int i = 0; i < rows; ++i) {
        const cv::Vec2f* inputRow = input_descentralized.ptr<cv::Vec2f>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = inputRow[j][0];
            in[idx][1] = inputRow[j][1];
        }
    }

    // Cria plano e executa a IFFT
    fftwf_plan p = fftwf_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // Armazenando apenas a parte real e normalizando
    cv::Mat ifft_img(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; ++i) {
        float* ifftRow = ifft_img.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            ifftRow[j] = out[idx][0] / (rows * cols);  // Normalizar
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    std::cout << "Saindo da IFFT com " << ifft_img.channels() << " canais." << std::endl;
    return ifft_img;
}

cv::Mat evm::applyDualLowPassFilter(const cv::Mat& dft_img, float lowCutoffFreq, float highCutoffFreq, float fps) const {
    std::cout << "applyDualLowPassFilter - Input channels: " << dft_img.channels() << std::endl;
    CV_Assert(dft_img.type() == CV_32FC2);

    int rows = dft_img.rows;
    int cols = dft_img.cols;
    int centerX = cols / 2;
    int centerY = rows / 2;

    float highFreqNorm = (rows / 2) * (highCutoffFreq / (fps / 2));
    float lowFreqNorm = (rows / 2) * (lowCutoffFreq / (fps / 2));

    cv::Mat mask(rows, cols, CV_32FC1, cv::Scalar(0));

    for (int i = 0; i < rows; i++) {
        float* maskRow = mask.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            float distX = j - centerX;
            float distY = i - centerY;
            float distance = sqrt(distX * distX + distY * distY);

            if (distance >= lowFreqNorm && distance <= highFreqNorm) {
                maskRow[j] = 1.0f;
            }
        }
    }

    // Aplicar a máscara ao DFT
    cv::Mat channels[2];
    cv::split(dft_img, channels);
    channels[0] = channels[0].mul(mask);
    channels[1] = channels[1].mul(mask);

    cv::Mat filtered;
    cv::merge(channels, 2, filtered);

    return filtered;
}

cv::Mat evm::ampImg(const cv::Mat& image, float alpha) const {
    cv::Mat amplified;
    image.convertTo(amplified, CV_32FC1, alpha);

    return amplified;
}

cv::Mat evm::pyr_up(const cv::Mat& img) const {
    cv::Mat img_up;
    cv::pyrUp(img, img_up);
    std::cout << "pyr_up - Output size: " << img_up.size() << ", channels: " << img_up.channels() << std::endl;
    return img_up;
}

cv::Mat evm::pyr_down(const cv::Mat& img) const {
    cv::Mat img_down;
    cv::pyrDown(img, img_down);
    std::cout << "pyr_down - Output size: " << img_down.size() << ", channels: " << img_down.channels() << std::endl;
    return img_down;
}

cv::Mat evm::processChannel(const cv::Mat& channel, float lowFreq, float highFreq, float fps, float alpha) {
    std::cerr << "Processando o canal em processChannel" << std::endl;
    cv::Mat channel_float;
    channel.convertTo(channel_float, CV_32FC1);

    // Aplicar redução de tamanho usando pyrDown antes da FFT
    cv::Mat reducedChannel = pyr_down(channel_float);

    std::cerr << "Aplicando a FFT" << std::endl;
    cv::Mat fft_result = applyFFT(reducedChannel);

    std::cerr << "Aplicando o filtro" << std::endl;
    cv::Mat filtered_result = applyDualLowPassFilter(fft_result, lowFreq, highFreq, fps);

    std::cerr << "Aplicando a IFFT" << std::endl;
    cv::Mat ifft_result = applyIFFT(filtered_result);

    std::cerr << "Amplificando a imagem" << std::endl;
    cv::Mat amplified = ampImg(ifft_result, alpha);

    // Aumentar a resolução para o tamanho original
    cv::Mat upsampled = pyr_up(amplified);

    // Garantir que o tamanho seja igual ao do canal original
    cv::resize(upsampled, upsampled, channel.size());

    // Combinar o canal original com o amplificado
    cv::Mat result;
    cv::add(channel_float, upsampled, result);

    // Normalizar e converter para o tipo original
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, channel.type());

    return result;
}

} // namespace my
