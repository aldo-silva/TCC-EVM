/**
 * @file ModelLoader.cpp
 * @brief Implementação da classe ModelLoader para carregar e executar modelos TensorFlow Lite.
 */

#include "ModelLoader.hpp"
#include <iostream>
#include "/home/aldo/tensorflow/tensorflow/lite/builtin_op_data.h"
#include "/home/aldo/tensorflow/tensorflow/lite/kernels/register.h"

#define INPUT_NORM_MEAN 127.5f
#define INPUT_NORM_STD  127.5f

/**
 * @brief Construtor da classe ModelLoader.
 * 
 * Inicializa o modelo a partir do caminho especificado, constrói o interpretador,
 * aloca tensores e preenche os tensores de entrada e saída.
 * 
 * @param modelPath Caminho para o arquivo do modelo TensorFlow Lite.
 */
my::ModelLoader::ModelLoader(std::string modelPath) {
    loadModel(modelPath.c_str());
    buildInterpreter();
    allocateTensors();
    fillInputTensors();
    fillOutputTensors();

    m_inputLoads.resize(getNumberOfInputs(), false);
}

/**
 * @brief Obtém a forma (shape) do tensor de entrada no índice especificado.
 * 
 * @param index Índice do tensor de entrada.
 * @return std::vector<int> Forma do tensor de entrada.
 */
std::vector<int> my::ModelLoader::getInputShape(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].dims;

    return std::vector<int>();
}

/**
 * @brief Obtém os dados do tensor de entrada no índice especificado.
 * 
 * @param index Índice do tensor de entrada.
 * @return float* Ponteiro para os dados do tensor de entrada.
 */
float* my::ModelLoader::getInputData(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].data;

    return nullptr;
}

/**
 * @brief Obtém o tamanho em bytes do tensor de entrada no índice especificado.
 * 
 * @param index Índice do tensor de entrada.
 * @return size_t Tamanho em bytes do tensor de entrada.
 */
size_t my::ModelLoader::getInputSize(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].bytes;

    return 0;
}

/**
 * @brief Obtém o número de tensores de entrada.
 * 
 * @return int Número de tensores de entrada.
 */
int my::ModelLoader::getNumberOfInputs() const {
    return m_inputs.size();
}

/**
 * @brief Obtém a forma (shape) do tensor de saída no índice especificado.
 * 
 * @param index Índice do tensor de saída.
 * @return std::vector<int> Forma do tensor de saída.
 */
std::vector<int> my::ModelLoader::getOutputShape(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].dims;
        
    return std::vector<int>();
}

/**
 * @brief Obtém os dados do tensor de saída no índice especificado.
 * 
 * @param index Índice do tensor de saída.
 * @return float* Ponteiro para os dados do tensor de saída.
 */
float* my::ModelLoader::getOutputData(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].data;

    return nullptr;
}

/**
 * @brief Obtém o tamanho em bytes do tensor de saída no índice especificado.
 * 
 * @param index Índice do tensor de saída.
 * @return size_t Tamanho em bytes do tensor de saída.
 */
size_t my::ModelLoader::getOutputSize(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].bytes;

    return 0;
}

/**
 * @brief Obtém o número de tensores de saída.
 * 
 * @return int Número de tensores de saída.
 */
int my::ModelLoader::getNumberOfOutputs() const {
    return m_outputs.size();
}

/**
 * @brief Carrega uma imagem para o tensor de entrada no índice especificado.
 * 
 * @param inputImage Imagem de entrada.
 * @param idx Índice do tensor de entrada.
 */
void my::ModelLoader::loadImageToInput(const cv::Mat& inputImage, int idx) {
    if (isIndexValid(idx, 'i')) {
        cv::Mat resizedImage = preprocessImage(inputImage, idx);
        loadBytesToInput(resizedImage.data, idx);
    }
}

/**
 * @brief Carrega bytes de dados para o tensor de entrada no índice especificado.
 * 
 * @param data Ponteiro para os dados a serem carregados.
 * @param idx Índice do tensor de entrada.
 */
void my::ModelLoader::loadBytesToInput(const void* data, int idx) {
    if (isIndexValid(idx, 'i')) {
        memcpy(m_inputs[idx].data, data, m_inputs[idx].bytes);
        m_inputLoads[idx] = true;
    }
}

/**
 * @brief Executa a inferência no modelo carregado.
 */
void my::ModelLoader::runInference() {
    inputChecker();
    m_interpreter->Invoke(); // Inferência do Tflite
}

/**
 * @brief Carrega os resultados da saída do modelo no índice especificado.
 * 
 * @param index Índice do tensor de saída.
 * @return std::vector<float> Resultados da inferência.
 */
std::vector<float> my::ModelLoader::loadOutput(int index) const {
    if (isIndexValid(index, 'o')) {
        int sizeInByte = m_outputs[index].bytes;
        int sizeInFloat = sizeInByte / sizeof(float);

        std::vector<float> inference(sizeInFloat);
        memcpy(&(inference[0]), m_outputs[index].data, sizeInByte);
        
        return inference;
    }
    return std::vector<float>();
}

//-------------------Métodos privados começam aqui-------------------

/**
 * @brief Carrega o modelo TensorFlow Lite a partir do caminho especificado.
 * 
 * @param modelPath Caminho para o arquivo do modelo.
 */
void my::ModelLoader::loadModel(const char* modelPath) {
    m_model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    if (m_model == nullptr) {
        std::cerr << "Falha ao construir FlatBufferModel a partir do arquivo: " << modelPath << std::endl;
        std::exit(1);
    }  
}

/**
 * @brief Constrói o interpretador do modelo TensorFlow Lite.
 * 
 * @param numThreads Número de threads para o interpretador.
 */
void my::ModelLoader::buildInterpreter(int numThreads) {
    tflite::ops::builtin::BuiltinOpResolver resolver;

    if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
        std::cerr << "Falha ao construir o interpretador." << std::endl;
        std::exit(1);
    }
    m_interpreter->SetNumThreads(numThreads);
}

/**
 * @brief Aloca os tensores do modelo.
 */
void my::ModelLoader::allocateTensors() {
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Falha ao alocar os tensores." << std::endl;
        std::exit(1);
    }
}

/**
 * @brief Preenche os tensores de entrada.
 */
void my::ModelLoader::fillInputTensors() {
    for (auto input: m_interpreter->inputs()) {
        TfLiteTensor* inputTensor =  m_interpreter->tensor(input);
        TfLiteIntArray* dims =  inputTensor->dims;

        m_inputs.push_back({
            inputTensor->data.f,
            inputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}

/**
 * @brief Preenche os tensores de saída.
 */
void my::ModelLoader::fillOutputTensors() {
    for (auto output: m_interpreter->outputs()) {
        TfLiteTensor* outputTensor =  m_interpreter->tensor(output);
        TfLiteIntArray* dims =  outputTensor->dims;

        m_outputs.push_back({
            outputTensor->data.f,
            outputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}

/**
 * @brief Verifica se o índice é válido para os tensores de entrada ou saída.
 * 
 * @param idx Índice a ser verificado.
 * @param c Tipo de tensor ('i' para entrada, 'o' para saída).
 * @return true Se o índice é válido.
 * @return false Se o índice não é válido.
 */
bool my::ModelLoader::isIndexValid(int idx, const char c) const {
    int size = 0;
    if (c == 'i')
        size = m_inputs.size();
    else if (c == 'o')
        size = m_outputs.size();
    else 
        return false;

    if (idx < 0 || idx >= size) {
        std::cerr << "Índice " << idx << " está fora do intervalo (" \
        << size << ")." << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Verifica se todos os tensores de entrada foram carregados.
 * 
 * @return true Se todos os tensores de entrada foram carregados.
 * @return false Se algum tensor de entrada não foi carregado.
 */
bool my::ModelLoader::isAllInputsLoaded() const {
    return (
        std::find(m_inputLoads.begin(), m_inputLoads.end(), false)
     == m_inputLoads.end()); 
}

/**
 * @brief Verifica se todos os tensores de entrada foram carregados e reseta o estado de carregamento.
 */
void my::ModelLoader::inputChecker() {
    if (isAllInputsLoaded() == false) {
        std::cerr << "Entrada ";
        for (int i = 0; i < m_inputLoads.size(); ++i) {
            if (m_inputLoads[i] == false) {
                std::cerr << i << " ";
            }
        }
        std::cerr << "não foi carregada." << std::endl;
        std::exit(1);
    }
    std::fill(m_inputLoads.begin(), m_inputLoads.end(), false);
}

/**
 * @brief Pré-processa a imagem de entrada para o formato esperado pelo modelo.
 * 
 * @param in Imagem de entrada.
 * @param idx Índice do tensor de entrada.
 * @return cv::Mat Imagem pré-processada.
 */
cv::Mat my::ModelLoader::preprocessImage(const cv::Mat& in, int idx) const {
    auto out = convertToRGB(in);

    std::vector<int> inputShape = getInputShape(idx);
    int H = inputShape[1];
    int W = inputShape[2]; 

    cv::Size wantedSize = cv::Size(W, H);
    cv::resize(out, out, wantedSize);

    /*
    Equivalente a (out - mean) / std
    */
    out.convertTo(out, CV_32FC3, 1 / INPUT_NORM_STD, -INPUT_NORM_MEAN / INPUT_NORM_STD);
    return out;
}

/**
 * @brief Converte a imagem de entrada para o formato RGB.
 * 
 * @param in Imagem de entrada.
 * @return cv::Mat Imagem convertida para RGB.
 */
cv::Mat my::ModelLoader::convertToRGB(const cv::Mat& in) const {
    cv::Mat out;
    int type = in.type();

    if (type == CV_8UC3) {
        cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
    }
    else if (type == CV_8UC4) {
        cv::cvtColor(in, out, cv::COLOR_BGRA2RGB);
    }
    else {
        std::cerr << "Tipo de imagem " << type << " não suportado" << std::endl;
        std::exit(1);
    }
    return out;
}

