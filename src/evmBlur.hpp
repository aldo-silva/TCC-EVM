#ifndef EVM_HPP
#define EVM_HPP

#include <opencv2/opencv.hpp>

namespace my {

class evm {
public:
    evm() = default;
    ~evm() = default;

    /**
     * Aplica um filtro de blur espacial para reduzir as altas frequências.
     * @param input Imagem de entrada.
     * @param ksize Tamanho do kernel do blur (por exemplo, 5 para um kernel 5x5).
     * @return Imagem filtrada.
     */
    cv::Mat applySpatialBlur(const cv::Mat& input, int ksize) const;

    /**
     * Aplica um filtro IIR band-pass simples para a filtragem temporal.
     * Neste exemplo, a implementação utiliza dois filtros IIR com parâmetros diferentes,
     * onde a subtração entre eles aproxima um filtro passa-banda.
     * @param input Imagem (ou frame) de entrada.
     * @param lowAlpha Parâmetro do filtro IIR para corte de frequências mais altas.
     * @param highAlpha Parâmetro do filtro IIR para corte de frequências mais baixas.
     * @return Imagem filtrada temporalmente.
     */
    cv::Mat applyBandPassIIR(const cv::Mat& input, float lowAlpha, float highAlpha);

    /**
     * Amplifica a variação do sinal.
     * @param image Imagem de entrada.
     * @param alpha Fator de amplificação.
     * @return Imagem amplificada.
     */
    cv::Mat ampImg(const cv::Mat& image, float alpha) const;

    /**
     * Reduz a imagem usando pirâmide gaussiana.
     * @param img Imagem de entrada.
     * @return Imagem reduzida.
     */
    cv::Mat pyr_down(const cv::Mat& img) const;

    /**
     * Amplia a imagem usando pirâmide gaussiana.
     * @param img Imagem de entrada.
     * @return Imagem ampliada.
     */
    cv::Mat pyr_up(const cv::Mat& img) const;

    /**
     * Processa um único canal conforme o fluxo:
     *   1. Reduz o tamanho da imagem (resize down).
     *   2. Aplica blur para reduzir altas frequências (filtro espacial).
     *   3. Aplica filtros IIR (passa-baixa) para filtrar temporalmente.
     *   4. Amplifica as variações.
     *   5. Retorna ao tamanho original (resize up) e adiciona o resultado ao canal original.
     * @param channel Canal de entrada.
     * @param alpha Fator de amplificação.
     * @return Canal processado.
     */
    cv::Mat processChannel(const cv::Mat& channel, float alpha);
};

} // namespace my

#endif // EVM_HPP
