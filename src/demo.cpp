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

            float widthFraction = 0.5f;
            float heightFraction = 0.25f;

            // Define the forehead region within the face ROI
            int foreheadWidth = static_cast<int>(roi.width * widthFraction);
            int foreheadHeight = static_cast<int>(roi.height * heightFraction);

            foreheadWidth -= foreheadWidth % 2;
            foreheadHeight -= foreheadHeight % 2;

            // if (foreheadWidth < 2) foreheadWidth = 2;
            // if (foreheadHeight < 2) foreheadHeight = 2;

            // Coordenadas iniciais para o ROI da testa
            int foreheadX = roi.x + (roi.width - foreheadWidth) / 2; // Centralizado horizontalmente no ROI do rosto
            int foreheadY = roi.y; // Início no topo do ROI do rosto

            int x = (roi.width - foreheadWidth) / 2; // Centered horizontally
            int y = 0; // Start at the top of the face ROI

            // cv::Rect foreheadRoi(x, y, foreheadWidth, foreheadHeight); // x=0, y=0 since it's relative to croppedFace

            cv::Rect foreheadRoi(foreheadX, foreheadY, foreheadWidth, foreheadHeight);

            foreheadRoi &= cv::Rect(0, 0, croppedFace.cols, croppedFace.rows);

            cv::rectangle(frame, foreheadRoi, cv::Scalar(255, 0, 0), 2);

            cv::Mat croppedForehead = croppedFace(foreheadRoi);

            cv::Mat ycrcb_forehead;
            cv::cvtColor(croppedForehead, ycrcb_forehead, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> ycrcb_channels;
            cv::split(ycrcb_forehead, ycrcb_channels);

            // Processar apenas o canal de crominância (Cr ou Cb)
            // std::cout << "Processando canal Cb da testa" << std::endl;
            cv::Mat processed_channel = evm_processor.processChannel(ycrcb_channels[2], lowFreq, highFreq, fps, alpha); // Usando o canal Cb

            // Substituir o canal processado na imagem
            ycrcb_channels[2] = processed_channel;
            cv::Mat processed_ycrcb;
            cv::merge(ycrcb_channels, processed_ycrcb);

            // Converter de volta para BGR para visualização
            cv::Mat processed_bgr;
            cv::cvtColor(processed_ycrcb, processed_bgr, cv::COLOR_YCrCb2BGR);

            processed_bgr.copyTo(croppedFace(foreheadRoi));

            // **Splitting RGB Channels After Processing**
            std::vector<cv::Mat> rgb_channels;
            cv::split(processed_bgr, rgb_channels);

            // Add frame data to signal processor
            signalProcessor.addFrameData(rgb_channels);

            // cv::imshow("Red Channel", rgb_channels[2]);
            // cv::imshow("Green Channel", rgb_channels[1]);
            // cv::imshow("Blue Channel", rgb_channels[0]);

            // croppedFace.copyTo(frame(roi));

            // Exibir a imagem processada
            // cv::imshow("Enhanced Face", processed_bgr);

            // Opcional: desenhar o ROI na imagem original
            // cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
            // cv::imshow("Face Detector", frame);

            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);

            // Compute heart rate and SpO₂ every time buffer is full
            if (signalProcessor.getGreenChannelMeans().size() == 300) { 
                double heartRate = signalProcessor.computeHeartRate(fps);
                std::cout << "Estimated Heart Rate: " << heartRate << " bpm" << std::endl;

                double spo2 = signalProcessor.computeSpO2();
                std::cout << "Estimated SpO₂: " << spo2 << "%" << std::endl;

                // Reset data after computation
                signalProcessor.reset();
            }


            // **Adicionar os valores de Heart Rate e SpO₂ na imagem**
            // Definir posição para os textos
            int baseLine = 0;
            cv::Size textSizeHR = cv::getTextSize("HR: ", cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
            cv::Size textSizeSpO2 = cv::getTextSize("SpO2: ", cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);

            cv::Point textOrgHR(roi.x, roi.y - 10); // Acima do ROI
            cv::Point textOrgSpO2(roi.x, roi.y - 30); // Acima do HR

            // Verificar se os valores foram calculados
            if (heartRate > 0.0) {
                std::string hrText = "HR: " + std::to_string(static_cast<int>(heartRate)) + " bpm";
                cv::putText(frame, hrText, textOrgHR, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            }

            if (spo2 > 0.0) {
                std::string spo2Text = "SpO2: " + std::to_string(static_cast<int>(spo2)) + " %";
                cv::putText(frame, spo2Text, textOrgSpO2, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }

        } else {
            cv::imshow("Face Detector", frame);
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