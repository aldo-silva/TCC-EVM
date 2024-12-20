#include "FaceDetection.hpp"
#include "evm.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "SignalProcessor.hpp"

#define SHOW_FPS (1)

#if SHOW_FPS
#include <chrono>
#endif

const std::string GSTREAMER_PIPELINE = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)360, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

int main(int argc, char* argv[]) {
    my::FaceDetection faceDetector("/home/aldo/Documentos/media_pipe-main/models");
    my::evm evm_processor;
    my::SignalProcessor signalProcessor;
    cv::VideoCapture cap(GSTREAMER_PIPELINE, cv::CAP_GSTREAMER);
    bool success = cap.isOpened();
    if (!success) {
        std::cerr << "Não foi possível abrir a câmera." << std::endl;
        return 1;
    }

#if SHOW_FPS
    float sum = 0;
    int count = 0;
    float fps = 0;
#endif

    // Definir as frequências de corte de acordo com o FPS esperado
    float lowFreq = 0.83f; // Frequência cardíaca mínima (~50 bpm)
    float highFreq = 3.0f; // Frequência cardíaca máxima (~180 bpm)
    float alpha = 50.0f;   // Fator de amplificação

    // Variáveis para armazenar os valores de Heart Rate e SpO₂
    double heartRate = 0.0;
    double spo2 = 0.0;

    while (true) {
        cv::Mat frame;
        success = cap.read(frame);
        if (!success) break;
        cv::flip(frame, frame, 1);

#if SHOW_FPS
        auto start = std::chrono::high_resolution_clock::now();
#endif

        faceDetector.loadImageToInput(frame);
        faceDetector.runInference();

#if SHOW_FPS
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float inferenceTime = duration.count() / 1e3;
        sum += inferenceTime;
        count += 1;
        fps = 1e3f / inferenceTime;
        cv::putText(frame, std::to_string(static_cast<int>(fps)) + " FPS", cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 196, 255), 2);
#endif

        cv::Rect roi = faceDetector.getFaceRoi();
        if (roi.width > 0 && roi.height > 0) {
            cv::Mat croppedFace = faceDetector.cropFrame(roi);

            // Definir proporções da testa em relação à face
            float widthFraction = 0.5f;
            float heightFraction = 0.25f;

            int foreheadWidth = static_cast<int>(roi.width * widthFraction);
            int foreheadHeight = static_cast<int>(roi.height * heightFraction);

            // Coordenadas do ROI da testa
            int foreheadX = roi.x + (roi.width - foreheadWidth) / 2; // Centralizado horizontalmente
            int foreheadY = roi.y; // No topo do ROI do rosto

            cv::Rect foreheadRoi(foreheadX, foreheadY, foreheadWidth, foreheadHeight);

            // Validar se o ROI está dentro do frame
            if (!croppedFace.empty() && foreheadRoi.width > 0 && foreheadRoi.height > 0) {
                foreheadRoi &= cv::Rect(0, 0, croppedFace.cols, croppedFace.rows);

                if (foreheadRoi.width > 0 && foreheadRoi.height > 0) {
                    cv::Mat croppedForehead = croppedFace(foreheadRoi);

                    // Verificar se o ROI da testa é válido
                    if (!croppedForehead.empty()) {
                        cv::Mat ycrcb_forehead;
                        cv::cvtColor(croppedForehead, ycrcb_forehead, cv::COLOR_BGR2YCrCb);

                        std::vector<cv::Mat> ycrcb_channels;
                        cv::split(ycrcb_forehead, ycrcb_channels);

                        // Processar o canal Cb da região da testa
                        cv::Mat processed_channel = evm_processor.processChannel(
                            ycrcb_channels[2], lowFreq, highFreq, fps, alpha);

                        // Substituir o canal processado
                        ycrcb_channels[2] = processed_channel;
                        cv::Mat processed_ycrcb;
                        cv::merge(ycrcb_channels, processed_ycrcb);

                        // Converter de volta para BGR e copiar para a face
                        cv::Mat processed_bgr;
                        cv::cvtColor(processed_ycrcb, processed_bgr, cv::COLOR_YCrCb2BGR);

                        processed_bgr.copyTo(croppedFace(foreheadRoi));

                        // Adicionar dados ao processador de sinais
                        std::vector<cv::Mat> rgb_channels;
                        cv::split(processed_bgr, rgb_channels);
                        signalProcessor.addFrameData(rgb_channels);
                    }
                }
            }

            // Desenhar o ROI da face e da testa
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);       // Verde: face
            cv::rectangle(frame, foreheadRoi, cv::Scalar(255, 0, 0), 2); // Azul: testa

            // Calcular frequência cardíaca e SpO₂ se o buffer estiver cheio
            if (signalProcessor.getGreenChannelMeans().size() == 300) { 
                heartRate = signalProcessor.computeHeartRate(fps);
                spo2 = signalProcessor.computeSpO2();

                std::cout << "Estimated Heart Rate: " << heartRate << " bpm" << std::endl;
                std::cout << "Estimated SpO₂: " << spo2 << "%" << std::endl;

                signalProcessor.reset(); // Resetar o buffer após o cálculo
            }

            // Adicionar informações na imagem
            cv::Point textOrgHR(roi.x, roi.y - 10);
            cv::Point textOrgSpO2(roi.x, roi.y - 30);

            if (heartRate > 0.0) {
                cv::putText(frame, "HR: " + std::to_string(static_cast<int>(heartRate)) + " bpm",
                            textOrgHR, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            }

            if (spo2 > 0.0) {
                cv::putText(frame, "SpO2: " + std::to_string(static_cast<int>(spo2)) + " %",
                            textOrgSpO2, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
        }

        cv::imshow("Face Detector", frame);

        if (cv::waitKey(10) == 27) break;  // Sair se 'ESC' for pressionado
    }

#if SHOW_FPS
    std::cout << "Average inference time: " << sum / count << " ms" << std::endl;
#endif

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
