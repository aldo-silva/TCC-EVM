#include "evm.hpp"
#include <iostream>
#include <fftw3.h>
#include <cmath>

namespace my {

// Função para trocar quadrantes (shift) no resultado da FFT
void fftShift(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    int cx = output.cols / 2;
    int cy = output.rows / 2;

    cv::Mat q0(output, cv::Rect(0, 0, cx, cy));   // Top-Left
    cv::Mat q1(output, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(output, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(output, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    // Swap quadrants (Top-Left <-> Bottom-Right)
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // Swap quadrants (Top-Right <-> Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

cv::Mat evm::applyFFT(const cv::Mat& input) const {
    CV_Assert(input.type() == CV_32FC1);

    int rows = input.rows;
    int cols = input.cols;

    fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);

    // Prepara dados para FFT
    for (int i = 0; i < rows; ++i) {
        const float* rowPtr = input.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = rowPtr[j]; // real
            in[idx][1] = 0.0f;      // imag
        }
    }

    fftwf_plan p = fftwf_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    cv::Mat dft_img(rows, cols, CV_32FC2);
    for (int i = 0; i < rows; ++i) {
        cv::Vec2f* dftRow = dft_img.ptr<cv::Vec2f>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            dftRow[j][0] = out[idx][0];
            dftRow[j][1] = out[idx][1];
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    // Centralizar o espectro
    cv::Mat shifted_dft;
    fftShift(dft_img, shifted_dft);

    return shifted_dft;
}

cv::Mat evm::applyIFFT(const cv::Mat& input) const {
    CV_Assert(input.type() == CV_32FC2);

    int rows = input.rows;
    int cols = input.cols;

    // Descentralizar antes da IFFT
    cv::Mat desc;
    fftShift(input, desc);

    fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);

    for (int i = 0; i < rows; ++i) {
        const cv::Vec2f* rowPtr = desc.ptr<cv::Vec2f>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = rowPtr[j][0];
            in[idx][1] = rowPtr[j][1];
        }
    }

    fftwf_plan p = fftwf_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // Armazenar apenas a parte real
    cv::Mat ifft_img(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; ++i) {
        float* rowPtr = ifft_img.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            // Normalizar dividindo por (rows*cols)
            rowPtr[j] = out[idx][0] / (rows * cols);
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    return ifft_img;
}

cv::Mat evm::applyDualLowPassFilter(const cv::Mat& dft_img,
                                    float lowCutoffFreq, float highCutoffFreq,
                                    float fps) const {
    CV_Assert(dft_img.type() == CV_32FC2);

    int rows = dft_img.rows;
    int cols = dft_img.cols;

    int centerX = cols / 2;
    int centerY = rows / 2;

    // Normalização frequencial para linhas/colunas
    float highFreqNorm = (rows / 2.0f) * (highCutoffFreq / (fps / 2.0f));
    float lowFreqNorm  = (rows / 2.0f) * (lowCutoffFreq  / (fps / 2.0f));

    // Criar máscara
    cv::Mat mask(rows, cols, CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < rows; i++) {
        float* maskRow = mask.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            float distX = (float)(j - centerX);
            float distY = (float)(i - centerY);
            float distance = std::sqrt(distX * distX + distY * distY);

            if (distance >= lowFreqNorm && distance <= highFreqNorm) {
                maskRow[j] = 1.0f;
            }
        }
    }

    // Aplicar a máscara ao DFT
    std::vector<cv::Mat> channels(2);
    cv::split(dft_img, channels);
    channels[0] = channels[0].mul(mask);
    channels[1] = channels[1].mul(mask);

    cv::Mat filtered;
    cv::merge(channels, filtered);

    return filtered;
}

cv::Mat evm::ampImg(const cv::Mat& image, float alpha) const {
    cv::Mat amplified;
    image.convertTo(amplified, CV_32FC1, alpha);
    return amplified;
}

cv::Mat evm::pyr_up(const cv::Mat& img) const {
    if (img.empty()) {
        std::cerr << "pyr_up called with empty image." << std::endl;
        return img;
    }
    cv::Mat img_up;
    cv::pyrUp(img, img_up);
    return img_up;
}

cv::Mat evm::pyr_down(const cv::Mat& img) const {
    if (img.empty()) {
        std::cerr << "pyr_down called with empty image." << std::endl;
        return img;
    }
    cv::Mat img_down;
    cv::pyrDown(img, img_down);
    return img_down;
}

cv::Mat evm::processChannel(const cv::Mat& channel, float lowFreq, float highFreq, float fps, float alpha) {
    // Converter para float
    cv::Mat channel_float;
    channel.convertTo(channel_float, CV_32FC1);

    if (channel_float.empty()) {
        std::cerr << "[processChannel] channel_float is empty. Returning original channel." << std::endl;
        return channel;
    }

    // Ajustar dimensões para serem pares
    int new_rows = channel_float.rows - (channel_float.rows % 2);
    int new_cols = channel_float.cols - (channel_float.cols % 2);

    if (new_rows <= 0 || new_cols <= 0) {
        std::cerr << "[processChannel] adjusted_channel has invalid size. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat adjusted_channel = channel_float(cv::Rect(0, 0, new_cols, new_rows));
    if (adjusted_channel.empty()) {
        std::cerr << "[processChannel] adjusted_channel is empty. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat reducedChannel = pyr_down(adjusted_channel);
    if (reducedChannel.empty()) {
        std::cerr << "[processChannel] reducedChannel is empty after pyr_down. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat fft_result = applyFFT(reducedChannel);
    if (fft_result.empty()) {
        std::cerr << "[processChannel] fft_result is empty. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat filtered_result = applyDualLowPassFilter(fft_result, lowFreq, highFreq, fps);
    if (filtered_result.empty()) {
        std::cerr << "[processChannel] filtered_result is empty. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat ifft_result = applyIFFT(filtered_result);
    if (ifft_result.empty()) {
        std::cerr << "[processChannel] ifft_result is empty. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat amplified = ampImg(ifft_result, alpha);
    if (amplified.empty()) {
        std::cerr << "[processChannel] amplified is empty. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat upsampled = pyr_up(amplified);
    if (upsampled.empty()) {
        std::cerr << "[processChannel] upsampled is empty after pyr_up. Returning original channel." << std::endl;
        return channel;
    }

    // Redimensionar para o tamanho de adjusted_channel
    if (upsampled.size() != adjusted_channel.size()) {
        cv::resize(upsampled, upsampled, adjusted_channel.size());
    }
    if (upsampled.empty()) {
        std::cerr << "[processChannel] upsampled is empty after resize. Returning original channel." << std::endl;
        return channel;
    }

    cv::Mat result;
    cv::add(adjusted_channel, upsampled, result);
    if (result.empty()) {
        std::cerr << "[processChannel] result is empty after cv::add. Returning original channel." << std::endl;
        return channel;
    }

    // Ajustar resultado para o tamanho original
    if (result.size() != channel.size()) {
        cv::resize(result, result, channel.size());
    }
    if (result.empty()) {
        std::cerr << "[processChannel] result is empty after final resize. Returning original channel." << std::endl;
        return channel;
    }

    // Normalizar e converter de volta
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, channel.type());

    return result;
}

} // namespace my
