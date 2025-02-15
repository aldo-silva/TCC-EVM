// demo.cpp
#include "FaceDetection.hpp"
#include "evm.hpp"
#include "SignalProcessor.hpp"
#include "Database.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define SHOW_FPS (1)

#if SHOW_FPS
#include <chrono>
#endif

const std::string VIDEO_FILE_PATH = "/home/aldo/Documentos/video/build/luz_natural_video_5s.avi";

int main(int argc, char* argv[]) {
    my::FaceDetection faceDetector("/home/aldo/Documentos/media_pipe-main/models");
    my::evm evm_processor;
    my::SignalProcessor signalProcessor;

    Database db;
    if (!db.open("/home/aldo/Documentos/TCC-EVM/server/measurement.db")) {
        std::cerr << "Falha ao abrir o banco de dados" << std::endl;
        return 1;
    }
    db.createTable();

    cv::VideoCapture cap(VIDEO_FILE_PATH);
    bool success = cap.isOpened();
    if (!success) {
        std::cerr << "Não foi possível abrir o arquivo de vídeo." << std::endl;
        return 1;
    }

#if SHOW_FPS
    float sum = 0;
    int count = 0;
    float fps = 0;
#endif

    float lowFreq = 0.83f;  // Frequência cardíaca mínima (~50 bpm)
    float highFreq = 3.0f;  // Frequência cardíaca máxima (~180 bpm)
    float alpha   = 50.0f;  // Fator de amplificação

    double heartRate = 0.0;
    double spo2 = 0.0;

    // Variáveis estáticas para "congelar" a detecção por 150 frames.
    static int frameCounter = 0;  
    static cv::Rect savedRoi(0, 0, 0, 0);  

    while (true) {
        cv::Mat frame;
        success = cap.read(frame);
        if (!success) break;

        cv::flip(frame, frame, 1);

#if SHOW_FPS
        auto start = std::chrono::high_resolution_clock::now();
#endif

        // Fazemos a detecção somente a cada 150 frames
        if (frameCounter % 150 == 0) {
            faceDetector.loadImageToInput(frame);
            faceDetector.runInference();
            savedRoi = faceDetector.getFaceRoi();  // Armazena o ROI detectado
        }
        frameCounter++;

#if SHOW_FPS
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float inferenceTime = duration.count() / 1e3;
        sum += inferenceTime;
        count += 1;
        fps = 1e3f / inferenceTime;
        cv::putText(frame, std::to_string(static_cast<int>(fps)) + " FPS",
                    cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 2,
                    cv::Scalar(0, 196, 255), 2);
#endif

        // Em vez de usar getFaceRoi() a cada frame, vamos usar o savedRoi
        cv::Rect roi = savedRoi;

        if (roi.width > 0 && roi.height > 0) {
            // Evita extrapolar a imagem
            roi &= cv::Rect(0, 0, frame.cols, frame.rows);

            cv::Mat croppedFace = faceDetector.cropFrame(roi);
            if (!croppedFace.empty()) {
                // ROI da testa
                float widthFraction      = 0.4f;
                float heightFraction     = 0.2f;
                float verticalOffsetFrac = 0.2f;

                int foreheadWidth  = static_cast<int>(roi.width * widthFraction);
                int foreheadHeight = static_cast<int>(roi.height * heightFraction);

                int foreheadX = (roi.width - foreheadWidth) / 2;
                int foreheadY = static_cast<int>(roi.height * verticalOffsetFrac);

                cv::Rect foreheadRoi(foreheadX, foreheadY, foreheadWidth, foreheadHeight);

                // Ajustar ao croppedFace (programação defensiva)
                foreheadRoi &= cv::Rect(0, 0, croppedFace.cols, croppedFace.rows);

                if (foreheadRoi.width > 1 && foreheadRoi.height > 1) {
                    cv::Mat croppedForehead = croppedFace(foreheadRoi);
                    if (!croppedForehead.empty()) {
                        // Conversão para YCrCb
                        cv::Mat ycrcb_forehead;
                        cv::cvtColor(croppedForehead, ycrcb_forehead, cv::COLOR_BGR2YCrCb);

                        std::vector<cv::Mat> ycrcb_channels;
                        cv::split(ycrcb_forehead, ycrcb_channels);

                        // Checamos o canal Cb (índice 2)
                        if (ycrcb_channels.size() == 3 && !ycrcb_channels[2].empty()) {
                            cv::Mat processed_channel = evm_processor.processChannel(
                                ycrcb_channels[2], lowFreq, highFreq, fps, alpha);

                            if (!processed_channel.empty()) {
                                // Substitui e converte de volta
                                ycrcb_channels[2] = processed_channel;
                                cv::Mat processed_ycrcb;
                                cv::merge(ycrcb_channels, processed_ycrcb);

                                cv::Mat processed_bgr;
                                cv::cvtColor(processed_ycrcb, processed_bgr, cv::COLOR_YCrCb2BGR);

                                // Ajusta tamanho, se necessário
                                if (processed_bgr.size() != croppedForehead.size()) {
                                    cv::resize(processed_bgr, processed_bgr, croppedForehead.size());
                                }
                                processed_bgr.copyTo(croppedFace(foreheadRoi));

                                // Atualiza o processador de sinais
                                std::vector<cv::Mat> rgb_channels;
                                cv::split(processed_bgr, rgb_channels);

                                if (rgb_channels.size() == 3) {
                                    signalProcessor.addFrameData(rgb_channels);
                                }
                            }
                        }
                    }
                }

                // Desenha retângulos (face e testa)
                cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
                cv::Point topLeft(roi.x + foreheadX, roi.y + foreheadY);
                cv::Point bottomRight(roi.x + foreheadX + foreheadWidth,
                                      roi.y + foreheadY + foreheadHeight);
                cv::rectangle(frame, cv::Rect(topLeft, bottomRight),
                              cv::Scalar(255, 0, 0), 2);

                // Quando o buffer tiver 150 amostras, calcular HR e SpO2
                if (signalProcessor.getGreenChannelMeans().size() == 150) {
                    auto t = std::time(nullptr);
                    auto tm = *std::localtime(&t);

                    char buffer[100];
                    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm);
                    std::string timestampStr(buffer);

                    heartRate = signalProcessor.computeHeartRate(fps, timestampStr);
                    spo2      = signalProcessor.computeSpO2();

                    std::cout << "Estimated Heart Rate: " << heartRate << " bpm" << std::endl;
                    std::cout << "Estimated SpO₂: " << spo2 << "%" << std::endl;

                    // Salva parâmetros intermediários
                    signalProcessor.saveIntermediateParameters(
                        "/home/aldo/data/spo2_intermediate_params" + timestampStr + ".csv");

                    std::string fileName = "/home/aldo/data/captures/" + timestampStr + ".png";
                    cv::imwrite(fileName, frame);

                    std::string relativePath = "captures/" + timestampStr + ".png";
                    db.insertMeasurement(heartRate, spo2, relativePath);

                    // Limpa o buffer de sinais
                    signalProcessor.reset();
                }

                // Mostra HR e SpO2 na tela
                cv::Point textOrgHR(roi.x, roi.y - 10);
                cv::Point textOrgSpO2(roi.x, roi.y - 30);

                if (heartRate > 0.0) {
                    cv::putText(frame,
                                "HR: " + std::to_string(static_cast<int>(heartRate)) + " bpm",
                                textOrgHR, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                cv::Scalar(255, 0, 0), 2);
                }
                if (spo2 > 0.0) {
                    cv::putText(frame,
                                "SpO2: " + std::to_string(static_cast<int>(spo2)) + " %",
                                textOrgSpO2, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        cv::imshow("Face Detector", frame);
        if (cv::waitKey(10) == 27) break; // ESC para sair
    }

#if SHOW_FPS
    std::cout << "Average inference time: " << sum / count << " ms" << std::endl;
#endif

    cap.release();
    cv::destroyAllWindows();
    db.close();
    return 0;
}
