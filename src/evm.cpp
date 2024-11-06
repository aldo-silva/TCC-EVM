#include "evm.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <complex>
#include <valarray>
#include <thread>
#include <vector>
#include <algorithm>
#include <fftw3.h>

using Complex = std::complex<float>;
using CArray = std::valarray<Complex>;

namespace my {



// Função para aplicar FFT usando Kiss FFT
cv::Mat evm::applyFFT(const cv::Mat& input) const {
    std::cout << "Entrando na FFT com " << input.channels() << " canais." << std::endl;
    CV_Assert(input.type() == CV_32FC1 || input.type() == CV_32FC2);

    int rows = input.rows;
    int cols = input.cols;

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // Prepara os dados para a FFT
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = input.at<cv::Vec2f>(i, j)[0];  // Parte real
            in[idx][1] = input.at<cv::Vec2f>(i, j)[1];  // Parte imaginária
        }
    }

    // Cria plano e executa a FFT
    p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    cv::Mat dft_img(rows, cols, CV_32FC2);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            dft_img.at<cv::Vec2f>(i, j)[0] = out[idx][0]; // Parte real
            dft_img.at<cv::Vec2f>(i, j)[1] = out[idx][1]; // Parte imaginária
        }
    }

    // Centralização dos dados da FFT (shifting quadrants)
    cv::Mat shifted_dft = dft_img.clone();
    int cx = cols / 2;
    int cy = rows / 2;

    // Troca quadrantes (Top-Left com Bottom-Right e Top-Right com Bottom-Left)
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            int new_i = (i + cy) % rows;
            int new_j = (j + cx) % cols;
            shifted_dft.at<cv::Vec2f>(new_i, new_j) = dft_img.at<cv::Vec2f>(i, j);
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    std::cout << "Saindo da FFT com " << shifted_dft.channels() << " canais." << std::endl;
    return shifted_dft;
}



cv::Mat evm::applyDualLowPassFilter(const cv::Mat& dft_img, float lowCutoffFreq, float highCutoffFreq, float fps) const {
    std::cout << "applyDualLowPassFilter - Input channels: " << dft_img.channels() << std::endl;
    CV_Assert(dft_img.channels() == 2); // Verifica se a imagem tem 2 canais

    cv::Mat dft_img_converted;
    dft_img.convertTo(dft_img_converted, CV_32FC2);

    if (dft_img_converted.empty()) {
        std::cerr << "Error: DFT image converted is empty." << std::endl;
        return cv::Mat();
    }

    int rows = dft_img_converted.rows;
    int cols = dft_img_converted.cols;
    int centerX = cols / 2;
    int centerY = rows / 2;

    float highFreqNorm = (rows / 2) * (highCutoffFreq / (fps / 2));
    float lowFreqNorm = (rows / 2) * (lowCutoffFreq / (fps / 2));

    cv::Mat mask(dft_img_converted.size(), CV_32FC2, cv::Scalar(0, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float distX = j - centerX;
            float distY = i - centerY;
            float distance = sqrt(distX * distX + distY * distY);

            if (distance < highFreqNorm) {
                float factor = 1.0f;
                if (distance > lowFreqNorm) {
                    float t = (distance - lowFreqNorm) / (highFreqNorm - lowFreqNorm);
                    factor = exp(-t * t);  // Aproximação gaussiana para suavizar a transição
                }
                mask.at<cv::Vec2f>(i, j) = cv::Vec2f(factor, factor);
            }
        }
    }

    cv::Mat mask_channels[2];  // Para armazenar canais real e imaginário
    cv::split(mask, mask_channels);  // Separa os canais real e imaginário

    cv::Mat mask_magnitude;
    cv::magnitude(mask_channels[0], mask_channels[1], mask_magnitude);  // Calcula a magnitude

    cv::normalize(mask_magnitude, mask_magnitude, 0, 255, cv::NORM_MINMAX);  // Normaliza para o intervalo visível
    mask_magnitude.convertTo(mask_magnitude, CV_8U);  // Converte para 8 bits unsigned
    cv::imshow("Máscara do Filtro", mask_magnitude);  // Mostra a imagem
    //cv::waitKey(0);  // Espera por uma tecla para continuar

    cv::Mat filtered;
    cv::mulSpectrums(dft_img_converted, mask, filtered, 0);
    return filtered;
}


cv::Mat evm::ampImg(const cv::Mat& inverseTransform) const {
cv::Mat amplified, scaled;
double minVal, maxVal;
cv::minMaxLoc(inverseTransform, &minVal, &maxVal);
std::cout << "Before amplification - Min: " << minVal << ", Max: " << maxVal << std::endl;

 if (inverseTransform.empty()) {
 std::cerr << "Error: inverseTransform is empty." << std::endl;
 return cv::Mat();
 }

 double ampFactor = 1;
 cv::multiply(inverseTransform, cv::Scalar(ampFactor), amplified);

cv::normalize(amplified, scaled, 0, 255, cv::NORM_MINMAX);
scaled.convertTo(scaled, CV_8U);

 cv::minMaxLoc(scaled, &minVal, &maxVal);
 std::cout << "After amplification - Min: " << minVal << ", Max: " << maxVal << std::endl;

return scaled;
}

cv::Mat evm::pyr_up(const cv::Mat& amp_img) const {
cv::Mat img_up;
cv::pyrUp(amp_img, img_up, cv::Size(amp_img.cols * 2, amp_img.rows * 2));
std::cout << "pyr_up - Output size: " << img_up.size() << ", channels: " << img_up.channels() << std::endl;
 return img_up;
}

cv::Mat evm::pyr_down(const cv::Mat& lab_img) const {
 cv::Mat img_down;
cv::pyrDown(lab_img, img_down, cv::Size(lab_img.cols / 2, lab_img.rows / 2));
std::cout << "pyr_down - Output size: " << img_down.size() << ", channels: " << img_down.channels() << std::endl;
return img_down;
}

// Função para aplicar IFFT usando Kiss FFT
cv::Mat evm::applyIFFT(const cv::Mat& input) const {
    std::cout << "Entrando na IFFT com " << input.channels() << " canais." << std::endl;
    CV_Assert(input.type() == CV_32FC2);

    int rows = input.rows;
    int cols = input.cols;

    // Descentralizar os dados antes da IFFT
    cv::Mat input_descentralized(input.size(), input.type());
    int cx = cols / 2;
    int cy = rows / 2;

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            int new_i = (i + cy) % rows;
            int new_j = (j + cx) % cols;
            input_descentralized.at<cv::Vec2f>(i, j) = input.at<cv::Vec2f>(new_i, new_j);
        }
    }

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // Prepara os dados para a IFFT
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = input_descentralized.at<cv::Vec2f>(i, j)[0];
            in[idx][1] = input_descentralized.at<cv::Vec2f>(i, j)[1];
        }
    }

    // Cria plano e executa a IFFT
    p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    cv::Mat ifft_img(rows, cols, CV_32FC1);  // Armazenando apenas a parte real
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            ifft_img.at<float>(i, j) = out[idx][0] / (rows * cols);  // Normalizar
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    std::cout << "Saindo da IFFT com " << ifft_img.channels() << " canais." << std::endl;
    return ifft_img;
}



// Atualiza a função processChannel para converter a imagem para o formato correto e usar FFT e IFFT

cv::Mat evm::processChannel(const cv::Mat& channel, float lowFreq, float highFreq, float fps) {
    std::cerr << "Processando o canal em processChannel" << std::endl;
    cv::Mat channel_converted, result;
    std::cout << "channel" << channel.size() << ", channels: " << channel.channels() << std::endl;
    std::cout << "channel_converted_1" << channel_converted.size() << ", channels: " << channel_converted.channels() << std::endl;
    // Converter o canal para o formato CV_32FC2 se não estiver

    if (channel.type() == CV_8UC1) {
        channel.convertTo(channel_converted, CV_32FC1);  // Converte para float
        channel_converted = cv::Mat_<cv::Vec2f>(channel_converted); // Converte para 2 canais
    } else if (channel.type() == CV_8UC3) {
        cv::cvtColor(channel, channel_converted, cv::COLOR_BGR2GRAY);
        channel_converted.convertTo(channel_converted, CV_32FC1);
        channel_converted = cv::Mat_<cv::Vec2f>(channel_converted); 
    } else {
        std::cerr << "Formato do canal não suportado." << std::endl;
        return cv::Mat();
    }
    std::cout << "channel_converted " << channel_converted.size() << ", channels: " << channel_converted.channels() << std::endl;
    // Aplicar redução de tamanho usando pyr_down antes de FFT
    cv::Mat reducedChannel = pyr_down(channel_converted);
    std::cerr << "Aplicando a FFT" << std::endl;
    cv::Mat fft_result = applyFFT(reducedChannel);

    cv::Mat filtered_result = applyDualLowPassFilter(fft_result, lowFreq, highFreq, fps);
    
    std::cerr << "Aplicando a IFFT" << std::endl;
    cv::Mat ifft_result = applyIFFT(filtered_result);
    cv::Mat amp_img = ampImg(ifft_result);
    cv::Mat upChannel = pyr_up(amp_img);
    std::cout << "upChannel " << upChannel.size() << ", channels: " << upChannel.channels() << std::endl;

    double alpha = 0.5, beta = 0.5, gamma = 0.0;
    std::cout << "channel_converted_3" << channel_converted.size() << ", channels: " << channel_converted.channels() << std::endl;
    cv::addWeighted(upChannel, alpha, channel_converted, beta, gamma, result);

    return result;
}

} // namespace my

