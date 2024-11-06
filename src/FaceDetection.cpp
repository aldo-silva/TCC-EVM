/**
 * @file FaceDetection.cpp
 * @brief Implementação das funções de detecção de faces utilizando um modelo de aprendizado de máquina.
 * 
 * Este arquivo contém a implementação das funções necessárias para carregar imagens,
 * executar a inferência e processar as detecções de faces.
 */

#include "FaceDetection.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <opencv2/imgproc.hpp>
/**
 * @brief Construtor da classe FaceDetection.
 * 
 * Inicializa o carregador de modelo com o diretório do modelo especificado.
 * 
 * @param modelDir Diretório contendo o modelo de detecção de faces.
 */
my::FaceDetection::FaceDetection(std::string modelDir) :
    my::ModelLoader(modelDir + std::string("/face_detection_short.tflite")) 
{}

/**
 * @brief Carrega uma imagem de entrada para o modelo.
 * 
 * @param in Imagem de entrada.
 * @param index Índice da entrada (padrão é 0).
 */
void my::FaceDetection::loadImageToInput(const cv::Mat& in, int index) {
    m_originImage = in;
    ModelLoader::loadImageToInput(in);
}

/**
 * @brief Executa a inferência no modelo carregado.
 * 
 * Executa a inferência, processa as saídas do regressor e do classificador,
 * e calcula a região de interesse (ROI) da face detectada.
 */
void my::FaceDetection::runInference() {
    ModelLoader::runInference();

    auto regressor = getFaceRegressor();
    auto classificator = getFaceClassificator();
    auto detection = m_postProcessor.getHighestScoreDetection(regressor, classificator);

    if (detection.classId != -1) {
        // A detecção ainda está na forma local [0..1]
        m_roi = calculateRoiFromDetection(detection);
    } else {
        m_roi = cv::Rect();
    }
}

/**
 * @brief Desenha a região de interesse (ROI) da face detectada no frame.
 * 
 * @param frame Frame no qual a ROI será desenhada.
 */
void my::FaceDetection::drawFaceROI(cv::Mat& frame) {
    // Verificar se um rosto foi detectado
    if (!m_roi.empty()) {
        // Desenhar um retângulo ao redor da ROI
        cv::rectangle(frame, m_roi, cv::Scalar(0, 255, 0), 2); // Definir cor e espessura do retângulo
    }
}

/**
 * @brief Obtém a imagem original carregada.
 * 
 * @return cv::Mat Imagem original.
 */
cv::Mat my::FaceDetection::getOriginalImage() const {
    return m_originImage;
}

/**
 * @brief Obtém os valores do regressor de faces a partir da saída do modelo.
 * 
 * @return std::vector<float> Vetor com os valores do regressor de faces.
 */
std::vector<float> my::FaceDetection::getFaceRegressor() const {
    return ModelLoader::loadOutput(0);
}

/**
 * @brief Obtém os valores do classificador de faces a partir da saída do modelo.
 * 
 * @return std::vector<float> Vetor com os valores do classificador de faces.
 */
std::vector<float> my::FaceDetection::getFaceClassificator() const {
    return ModelLoader::loadOutput(1);
}

/**
 * @brief Obtém a região de interesse (ROI) da face detectada.
 * 
 * @return cv::Rect ROI da face detectada.
 */
cv::Rect my::FaceDetection::getFaceRoi() const {
    return m_roi;
}

/**
 * @brief Recorta um frame com base na região de interesse (ROI) fornecida.
 * 
 * @param roi Região de interesse utilizada para recortar o frame.
 * @return cv::Mat Frame recortado.
 */
cv::Mat my::FaceDetection::cropFrame(const cv::Rect& roi) const {
    cv::Mat frame = getOriginalImage();
    if (frame.empty()) {
        return cv::Mat();
    }

    int x = std::max(0, roi.x);
    int y = std::max(0, roi.y);
    int width = std::min(roi.width, frame.cols - x);
    int height = std::min(roi.height, frame.rows - y);

    cv::Rect safeRoi(x, y, width, height);
    cv::Mat croppedFace = frame(safeRoi);

    // Redimensionar para a próxima potência de dois
    int newWidth = pow(2, ceil(log2(croppedFace.cols)));
    int newHeight = pow(2, ceil(log2(croppedFace.rows)));
    cv::resize(croppedFace, croppedFace, cv::Size(newWidth, newHeight));

    return croppedFace;
}



//-------------------Métodos privados começam aqui-------------------

/**
 * @brief Calcula a região de interesse (ROI) a partir da detecção.
 * 
 * @param detection Detecção utilizada para calcular a ROI.
 * @return cv::Rect ROI calculada.
 */
cv::Rect my::FaceDetection::calculateRoiFromDetection(const Detection& detection) const {
    int origWidth = m_originImage.size().width;
    int origHeight = m_originImage.size().height;
    
    auto center = (detection.roi.tl() + detection.roi.br()) * 0.5f;
    center.x *= origWidth;
    center.y *= origHeight;

    auto w = detection.roi.width * origWidth * 1.5f;
    auto h = detection.roi.height * origHeight * 2.f;

    return cv::Rect((int)center.x - w/2, (int)center.y - h/2, (int)w, (int)h);
}


