/**
 * @file FaceDetection.hpp
 * @brief Definições da classe FaceDetection para utilização do Mediapipe Face Detector.
 * 
 * Este arquivo contém as definições da classe FaceDetection, que é um wrapper do modelo
 * de detecção de faces do Mediapipe.
 */

#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include "ModelLoader.hpp"
#include "DetectionPostProcess.hpp"

namespace my {

    /**
     * @brief Classe que encapsula o uso do Mediapipe Face Detector.
     * 
     * Esta classe não pode ser copiada.
     */
    class FaceDetection : public my::ModelLoader {
        public:
            /**
             * @brief Construtor da classe FaceDetection.
             * 
             * O usuário deve fornecer o diretório contendo o arquivo face_detection_short.tflite,
             * e não o próprio arquivo.
             * 
             * @param modelPath Caminho do diretório contendo o modelo de detecção de faces.
             */
            FaceDetection(std::string modelPath);
            
            /**
             * @brief Destrutor padrão da classe FaceDetection.
             */
            virtual ~FaceDetection() = default;

            /**
             * @brief Obtém a imagem de entrada original.
             * 
             * @return cv::Mat Imagem de entrada original.
             */
            cv::Mat getOriginalImage() const;

            /**
             * @brief Obtém o resultado do regressor (primeiro tensor de saída).
             * 
             * @return std::vector<float> Resultado do regressor.
             */
            std::vector<float> getFaceRegressor() const;

            /**
             * @brief Obtém o resultado do classificador (segundo tensor de saída).
             * 
             * @return std::vector<float> Resultado do classificador.
             */
            std::vector<float> getFaceClassificator() const;

            /**
             * @brief Obtém a posição da face com maior confiança.
             * 
             * A posição é relativa à imagem passada para InputTensor(0).
             * 
             * @return cv::Rect Posição da face detectada.
             */
            virtual cv::Rect getFaceRoi() const;

            /**
             * @brief Carrega uma imagem de entrada para o tensor de entrada do modelo.
             * 
             * @param inputImage Imagem de entrada.
             * @param index Índice da entrada (não é relevante, o modelo sempre carrega para InputTensor(0)).
             */
            virtual void loadImageToInput(const cv::Mat& inputImage, int index = 0);

            /**
             * @brief Executa a inferência no modelo carregado.
             * 
             * Pode ser executado somente quando todos os tensores de entrada foram carregados.
             */
            virtual void runInference();

            /**
             * @brief Recorta o frame de entrada na região de interesse (ROI), adicionando padding se necessário.
             * 
             * @param roi Região de interesse para recortar o frame.
             * @return cv::Mat Frame recortado.
             */
            cv::Mat cropFrame(const cv::Rect& roi) const;

            /**
             * @brief Desenha a região de interesse (ROI) da face detectada no frame.
             * 
             * @param frame Frame no qual a ROI será desenhada.
             */
            void drawFaceROI(cv::Mat& frame);

        private:
            /**
             * @brief Sobrescreve a função loadBytesToInput da classe base ModelLoader.
             * 
             * Esta classe pode carregar apenas imagens para a entrada.
             */
            using ModelLoader::loadBytesToInput;

            /**
             * @brief Converte a caixa de detecção de volta ao tamanho original.
             * 
             * @param detection Detecção utilizada para calcular a ROI.
             * @return cv::Rect ROI calculada.
             */
            cv::Rect calculateRoiFromDetection(const Detection& detection) const;

        private:
            /**
             * @brief Auxilia na obtenção da região de interesse a partir das saídas do modelo.
             */
            DetectionPostProcess m_postProcessor;

            /**
             * @brief Armazena algumas informações.
             */
            cv::Mat m_originImage;

            /**
             * @brief Região de interesse (ROI) da face detectada.
             */
            cv::Rect m_roi;
    };
}

#endif // FACEDETECTION_H

