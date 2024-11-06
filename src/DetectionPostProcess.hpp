/**
 * @file DetectionPostProcess.hpp
 * @brief Definições de estruturas e classes para pós-processamento de detecções.
 * 
 * Este arquivo contém as definições de estruturas e classes necessárias para 
 * converter a saída da detecção de faces do Mediapipe em caixas delimitadoras de faces.
 */

#ifndef DETECTIONPOSTPROCESS_H
#define DETECTIONPOSTPROCESS_H

#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include "opencv2/core.hpp"

#define CLASS_ID        0
#define MIN_THRESHOLD   0.75f
#define DETECTION_SIZE  128
#define NUM_BOXES       896
#define NUM_COORD       16
#define NUM_SIZES       2

namespace my {

    /**
     * @brief Estrutura que define as opções de âncoras.
     * 
     * Esta estrutura contém os tamanhos e número de camadas para gerar âncoras,
     * bem como os offsets para o centro das âncoras.
     */
    struct AnchorOptions {
        const int sizes[NUM_SIZES] = {16, 8};        ///< Tamanhos das âncoras.
        const int numLayers[NUM_SIZES] = {2, 6};     ///< Número de camadas de âncoras.

        const float offsetX = 0.5f;                  ///< Offset X para o centro das âncoras.
        const float offsetY = 0.5f;                  ///< Offset Y para o centro das âncoras.
    };

    /**
     * @brief Estrutura que define uma detecção.
     * 
     * Esta estrutura contém a caixa delimitadora (ROI), a pontuação e o ID da classe da detecção.
     */
    struct Detection {
        cv::Rect2f roi;  ///< Caixa delimitadora da detecção.
        float score;     ///< Pontuação da detecção.
        int classId;     ///< ID da classe da detecção.

        /**
         * @brief Construtor padrão da estrutura Detection.
         */
        Detection() : score(), classId(-1), roi() {}

        /**
         * @brief Construtor da estrutura Detection com parâmetros.
         * 
         * @param score Pontuação da detecção.
         * @param classId ID da classe da detecção.
         * @param roi Caixa delimitadora da detecção.
         */
        Detection(float score, int classId, cv::Rect2f roi) :
            score(score), classId(classId), roi(roi) {}
        
        /**
         * @brief Destrutor padrão da estrutura Detection.
         */
        ~Detection() = default;
    };

    /**
     * @brief Classe auxiliar que converte a saída da detecção de faces do Mediapipe em caixas delimitadoras de faces.
     */
    class DetectionPostProcess {
        public:
            /**
             * @brief Construtor da classe DetectionPostProcess.
             * 
             * Inicializa as âncoras utilizando as opções padrão.
             */
            DetectionPostProcess();

            /**
             * @brief Destrutor padrão da classe DetectionPostProcess.
             */
            ~DetectionPostProcess() = default;

            /**
             * @brief Obtém a detecção com a maior pontuação.
             * 
             * @param rawBoxes Vetor contendo os valores brutos das caixas.
             * @param scores Vetor contendo as pontuações das detecções.
             * @return Detection Detecção com a maior pontuação.
             */
            Detection getHighestScoreDetection
            (const std::vector<float>& rawBoxes, const std::vector<float>& scores) const;

        private:
            /**
             * @brief Decodifica uma caixa delimitadora a partir dos valores brutos.
             * 
             * @param rawBoxes Vetor contendo os valores brutos das caixas.
             * @param index Índice da caixa a ser decodificada.
             * @return cv::Rect2f Caixa delimitadora decodificada.
             */
            cv::Rect2f decodeBox(const std::vector<float>& rawBoxes, int index) const;

        private:
            std::vector<cv::Rect2f> m_anchors; ///< Vetor de âncoras geradas.
    };
}

#endif // DETECTIONPOSTPROCESS_H

