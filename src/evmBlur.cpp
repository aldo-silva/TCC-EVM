#include "evmBlur.hpp"
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace my {

/**
 * Exemplo simples de um IIR de 1ª ordem:
 *   out[n] = (1 - alpha)*out[n-1] + alpha*in[n]
 * Geralmente, para um band-pass, você teria dois filtros (low-pass + high-pass).
 */
class IIRFilter {
public:
    IIRFilter(float alpha)
        : alpha_(alpha)
    {}

    cv::Mat process(const cv::Mat& input) {
        if (prev_.empty()) {
            input.copyTo(prev_);
            return input.clone();
        }
        // out = (1 - alpha)*prev + alpha*input
        cv::Mat output;
        output = prev_ * (1.0f - alpha_) + input * alpha_;
        output.copyTo(prev_);
        return output;
    }

private:
    float alpha_;
    cv::Mat prev_;
};

/**
 * Função simples de blur (caixa) como substituto do filtro espacial via FFT.
 * Você pode trocar por GaussianBlur ou outro filtro se preferir.
 */
cv::Mat evm::applySpatialBlur(const cv::Mat& input, int ksize) const {
    CV_Assert(!input.empty());

    cv::Mat output;
    cv::blur(input, output, cv::Size(ksize, ksize));
    return output;
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

/**
 * Exemplo de combinação de 2 filtros IIR para simular um band-pass:
 *  - iirLow  = low-pass  para ~ 240 BPM (parte superior da banda)
 *  - iirHigh = low-pass  para ~ 40  BPM (depois subtraímos do sinal original para criar um high-pass)
 *
 * Obs.: a implementação exata de band-pass com IIR depende de como você define suas constantes
 *       e de como converte BPM -> Frequências -> Ganhos do filtro.
 *       Abaixo é apenas um rascunho de exemplo.
 */
cv::Mat evm::applyBandPassIIR(const cv::Mat& current, IIRFilter& iirLow, IIRFilter& iirHigh) const
{
    // Filtro passa-baixa mais agressivo (para recortar frequências muito altas)
    cv::Mat lowPassed = iirLow.process(current);

    // Filtro passa-baixa para recortar frequências abaixo do mínimo
    // (quando subtraímos do sinal original, obtemos as frequências acima desse limite)
    cv::Mat lowPassed2 = iirHigh.process(current);

    // Band-pass aproximado: parte do sinal entre as duas faixas
    //   bandPassed = (frequências até upper) - (frequências até lower)
    //   Em termos práticos,: bandPassed = lowPassed - lowPassed2
    cv::Mat bandPassed;
    cv::subtract(lowPassed, lowPassed2, bandPassed);

    return bandPassed;
}

/**
 * Versão adaptada do processChannel que:
 * 1) Reduz a imagem
 * 2) Aplica blur (no lugar de FFT)
 * 3) Aplica filtragem temporal (exemplo IIR band-pass)
 * 4) Amplifica
 * 5) Retorna ao tamanho original e adiciona ao canal
 */
cv::Mat evm::processChannel(const cv::Mat& channel, float alpha)

{
    // Converter para float [0..255]
    cv::Mat channel_float;
    channel.convertTo(channel_float, CV_32FC1);

    // Garantir dimensões pares para o pyrDown
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

    // 1) Reduz a imagem
    cv::Mat reducedChannel = pyr_down(adjusted_channel);

    // 2) Aplica o blur (filtro espacial para reduzir altas frequências)
    cv::Mat blurred = applySpatialBlur(reducedChannel, 5); // ksize=5 como exemplo

    // 3) Exemplo de filtros IIR para simular um band-pass. Ajuste alpha(s) conforme necessário
    static IIRFilter iirLow(0.1f);   // alpha do low-pass 1
    static IIRFilter iirHigh(0.01f); // alpha do low-pass 2 (mais lento)
    cv::Mat bandPassed = applyBandPassIIR(blurred, iirLow, iirHigh);

    // 4) Amplifica a variação
    cv::Mat amplified = ampImg(bandPassed, alpha);

    // 5) Retorna ao tamanho original
    // 5.1) Primeiro retorna ao tamanho do "adjusted_channel"
    cv::Mat upsampled = pyr_up(amplified);
    if (upsampled.size() != adjusted_channel.size()) {
        cv::resize(upsampled, upsampled, adjusted_channel.size());
    }

    // 5.2) Soma a este canal reduzido original
    cv::Mat result;
    cv::add(adjusted_channel, upsampled, result);

    // 5.3) Ajusta para o tamanho do canal de entrada (pode estar maior)
    if (result.size() != channel.size()) {
        cv::resize(result, result, channel.size());
    }

    // Normaliza e converte de volta
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, channel.type());

    return result;
}

} // namespace my
