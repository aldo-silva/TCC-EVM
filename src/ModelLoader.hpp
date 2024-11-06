/**
 * @file FaceLoader.hpp
 * @brief Definições da classe ModelLoadert para utilização do Mediapipe Face Detector.
 * 
 * Este arquivo contém as definições da classe ModelLoader, que é um wrapper do modelo
 * de detecção de faces do Mediapipe.
 */


#ifndef MODELLOADER_H
#define MODELLOADER_H

#include <vector>
#include <memory>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "/home/aldo/tensorflow/tensorflow/lite/interpreter.h"
#include "/home/aldo/tensorflow/tensorflow/lite/model.h"

namespace my {

    /**
     * @brief Tipo matriz usando std::vector para armazenamento.
     */
    template <class T>
    using Matrix = std::vector<std::vector<T>>;

    /**
     * @brief Um wrapper para salvar informações dos tensores do TFLite.
     */
    struct TensorWrapper {
        float* data;                ///< Ponteiro para os dados do tensor.
        size_t bytes;               ///< Tamanho dos dados em bytes.
        std::vector<int> dims;      ///< Forma do tensor.

        /**
         * @brief Construtor para TensorWrapper.
         * @param t_data Ponteiro para os dados do tensor.
         * @param t_bytes Tamanho dos dados em bytes.
         * @param t_dims Ponteiro para a forma do tensor.
         * @param t_dimSize Tamanho da forma do tensor.
         */
        TensorWrapper(float* t_data, size_t t_bytes, int* t_dims, int t_dimSize): 
            data(t_data), bytes(t_bytes), dims(t_dims, t_dims + t_dimSize) {}
    };

    /**
     * @brief Um wrapper para simplificar o uso de modelos do TFLite.
     * 
     * Esta classe não pode ser copiada.
     */
    class ModelLoader {
        public:
            /**
             * @brief Construtor a partir de um arquivo .tflite.
             * @param modelPath Caminho para o arquivo .tflite.
             */
            ModelLoader(std::string modelPath);
            ModelLoader(const ModelLoader& other) = delete;
            ModelLoader& operator=(const ModelLoader& other) = delete;
            virtual ~ModelLoader() = default;

            /**
             * @brief Obtém a forma do tensor de entrada no índice especificado.
             * @param index Índice do tensor de entrada.
             * @return std::vector<int> Forma do tensor de entrada.
             */
            std::vector<int> getInputShape(int index = 0) const;

            /**
             * @brief Obtém o ponteiro para os dados do tensor de entrada no índice especificado.
             * @param index Índice do tensor de entrada.
             * @return float* Ponteiro para os dados do tensor de entrada.
             */
            float* getInputData(int index = 0) const;

            /**
             * @brief Obtém o tamanho em bytes do tensor de entrada no índice especificado.
             * @param index Índice do tensor de entrada.
             * @return size_t Tamanho em bytes do tensor de entrada.
             */
            size_t getInputSize(int index = 0) const;

            /**
             * @brief Obtém o número de tensores de entrada necessários para executar a inferência.
             * @return int Número de tensores de entrada.
             */
            int getNumberOfInputs() const;

            /**
             * @brief Obtém a forma do tensor de saída no índice especificado.
             * @param index Índice do tensor de saída.
             * @return std::vector<int> Forma do tensor de saída.
             */
            std::vector<int> getOutputShape(int index = 0) const;

            /**
             * @brief Obtém o ponteiro para os dados do tensor de saída no índice especificado.
             * @param index Índice do tensor de saída.
             * @return float* Ponteiro para os dados do tensor de saída.
             */
            float* getOutputData(int index = 0) const;

            /**
             * @brief Obtém o tamanho em bytes do tensor de saída no índice especificado.
             * @param index Índice do tensor de saída.
             * @return size_t Tamanho em bytes do tensor de saída.
             */
            size_t getOutputSize(int index = 0) const;

            /**
             * @brief Obtém o número de tensores de saída da inferência.
             * @return int Número de tensores de saída.
             */
            int getNumberOfOutputs() const;

            /**
             * @brief Carrega uma imagem (formato BGR) no modelo no índice especificado.
             * 
             * Suporta apenas imagens dos tipos CV_8UC3 e CV_8UC4.
             * @param inputImage Imagem de entrada.
             * @param index Índice do tensor de entrada.
             */
            virtual void loadImageToInput(const cv::Mat& inputImage, int index = 0);

            /**
             * @brief Carrega dados em bytes no modelo no índice especificado.
             * @param data Ponteiro para os dados a serem carregados.
             * @param index Índice do tensor de entrada.
             */
            virtual void loadBytesToInput(const void* data, int index = 0);

            /**
             * @brief Executa a inferência nos dados de entrada.
             * 
             * Só pode ser executado quando todos os tensores de entrada foram carregados.
             */
            virtual void runInference();

            /**
             * @brief Obtém os dados de saída da inferência no índice especificado.
             * 
             * A forma dos dados é achatada a partir de getOutputShape(index).
             * @param index Índice do tensor de saída.
             * @return std::vector<float> Dados de saída da inferência.
             */
            virtual std::vector<float> loadOutput(int index = 0) const;

        private:
            /**
             * @brief Funções auxiliares do construtor.
             */
            void loadModel(const char* modelPath);
            void buildInterpreter(int numThreads = -1);
            void allocateTensors();           
            void fillInputTensors();
            void fillOutputTensors();

            /**
             * @brief Verifica se o índice é válido para o tensor de entrada ou saída.
             * @param index Índice a ser verificado.
             * @param c Tipo de tensor ('i' para entrada, 'o' para saída).
             * @return true Se o índice é válido.
             * @return false Se o índice não é válido.
             */
            bool isIndexValid(int index, const char c = 'i') const;

            /**
             * @brief Verifica se todos os tensores de entrada foram carregados.
             * @return true Se todos os tensores de entrada foram carregados.
             * @return false Se algum tensor de entrada não foi carregado.
             */
            bool isAllInputsLoaded() const;

            /**
             * @brief Processa as cargas de entrada antes de executar a inferência.
             */
            void inputChecker();

            /**
             * @brief Converte a imagem para float e redimensiona para a forma de getInputShape(idx).
             * @param in Imagem de entrada.
             * @param idx Índice do tensor de entrada.
             * @return cv::Mat Imagem pré-processada.
             */
            cv::Mat preprocessImage(const cv::Mat& in, int idx) const;

            /**
             * @brief Converte a imagem do tipo CV_8UC3 ou CV_8UC4 para o formato RGB.
             * @param in Imagem de entrada.
             * @return cv::Mat Imagem convertida para RGB.
             */
            cv::Mat convertToRGB(const cv::Mat& in) const;

        private:
            std::vector<TensorWrapper> m_inputs;          ///< Informações dos tensores de entrada.
            std::vector<TensorWrapper> m_outputs;         ///< Informações dos tensores de saída.
            std::unique_ptr<tflite::FlatBufferModel> m_model; ///< Modelo TFLite.
            std::unique_ptr<tflite::Interpreter> m_interpreter; ///< Interpretador TFLite.
            std::vector<bool> m_inputLoads;               ///< Rastreamento dos tensores de entrada carregados.
    };

} // namespace my

#endif // MODELLOADER_H

