#include "FaceDetection.hpp"
#include "evm.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define SHOW_FPS (1)

#if SHOW_FPS
#include <chrono>
#endif

const std::string GSTREAMER_PIPELINE = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)360, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

int main(int argc, char* argv[]) {
    my::FaceDetection faceDetector("/home/aldo/Documentos/media_pipe-main/models");
    my::evm evm_processor;
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

            // Converter para o espaço de cor YCrCb para separar luminância de crominância
            cv::Mat ycrcb_face;
            cv::cvtColor(croppedFace, ycrcb_face, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> ycrcb_channels;
            cv::split(ycrcb_face, ycrcb_channels);

            // Processar apenas o canal de crominância (Cr ou Cb)
            std::cout << "Processando canal Cb" << std::endl;
            cv::Mat processed_channel = evm_processor.processChannel(ycrcb_channels[2], lowFreq, highFreq, fps, alpha); // Usando o canal Cb

            // Substituir o canal processado na imagem
            ycrcb_channels[2] = processed_channel;
            cv::Mat processed_ycrcb;
            cv::merge(ycrcb_channels, processed_ycrcb);

            // Converter de volta para BGR para visualização
            cv::Mat processed_bgr;
            cv::cvtColor(processed_ycrcb, processed_bgr, cv::COLOR_YCrCb2BGR);

            // Exibir a imagem processada
            cv::imshow("Enhanced Face", processed_bgr);

            // Opcional: desenhar o ROI na imagem original
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Face Detector", frame);
        } else {
            cv::imshow("Face Detector", frame);
        }

        if (cv::waitKey(10) == 27) break;  // Sair se 'ESC' for pressionado
    }

#if SHOW_FPS
    std::cout << "Average inference time: " << sum / count << " ms" << std::endl;
#endif

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
