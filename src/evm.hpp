#ifndef EVM_H
#define EVM_H

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "fftw3.h"
namespace my {

/**
 * @class evm
 * @brief Classe que implementa vários métodos para processamento de vídeo e imagem utilizando transformadas de frequência.
 */
class evm
{
public:
    /**
     * @brief Reduz a resolução da imagem usando pyramidal down.
     * @param lab_img Imagem no espaço de cor LAB.
     * @return Imagem com resolução reduzida.
     */
    cv::Mat pyr_down(const cv::Mat& lab_img) const;

    /**
     * @brief Aplica a Transformada Rápida de Fourier (FFT) em uma imagem.
     * @param channel Imagem ou canal da imagem.
     * @return Imagem no domínio da frequência.
     */
	 cv::Mat applyFFT(const cv::Mat& input) const;

    /**
     * @brief Aplica um filtro passa-banda em uma imagem no domínio da frequência.
     * @param dft_img Imagem no domínio da frequência resultante de uma FFT.
     * @param lowFreq Frequência baixa do filtro.
     * @param highFreq Frequência alta do filtro.
     * @param fps Frames por segundo do vídeo (usado para cálculo das frequências).
     * @return Imagem filtrada no domínio da frequência.
     */
    cv::Mat applyDualLowPassFilter(const cv::Mat& dft_img, float lowFreq, float highFreq, float fps) const;
    //cv::Mat applyBandpassFilter(const cv::Mat& dft_img, float lowFreq, float highFreq, float fps);
    /**
     * @brief Realiza a transformada inversa de Fourier para converter a imagem de volta ao domínio do tempo.
     * @param filtered Imagem filtrada no domínio da frequência.
     * @return Imagem no domínio do tempo.
     */
    cv::Mat applyIFFT(const cv::Mat& input) const;

    /**
     * @brief Amplifica a imagem por um fator de 100.
     * @param inverseTransform Imagem resultante da transformada inversa de Fourier.
     * @return Imagem amplificada.
     */
    cv::Mat ampImg(const cv::Mat& inverseTransform) const;

    /**
     * @brief Aumenta a resolução da imagem usando pyramidal up.
     * @param amp_img Imagem amplificada.
     * @return Imagem com resolução aumentada.
     */
    cv::Mat pyr_up(const cv::Mat& amp_img) const;

    /**
     * @brief Processa um canal da imagem aplicando as transformações necessárias.
     * @param channel Canal da imagem a ser processado.
     * @param lowFreq Frequência baixa do filtro.
     * @param highFreq Frequência alta do filtro.
     * @param fps Frames por segundo do vídeo (usado para cálculo das frequências).
     * @return Imagem processada.
     */
    cv::Mat processChannel(const cv::Mat& channel, float lowFreq, float highFreq, float fps);
};

}

#endif // EVM_H


