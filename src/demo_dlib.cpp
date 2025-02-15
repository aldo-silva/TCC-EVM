#include "evm.hpp"
#include "SignalProcessor.hpp"
#include "Database.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/dnn.h>                // Para a rede CNN
#include <dlib/image_processing.h>   // Para shape_predictor
#include <dlib/opencv.h>
#include <dlib/image_io.h>
// #include <dlib/image_processing/frontal_face_detector.h> // Não precisamos mais desse cabeçalho

#define SHOW_FPS (1)

#if SHOW_FPS
#include <chrono>
#endif

// Caminho do vídeo
const std::string VIDEO_FILE_PATH = "/home/aldo/Documentos/video/build/luz_natural_video_5s.avi";

// ----------------------------------------------------------
// Definição da rede CNN conforme o exemplo oficial da dlib
// (dnn_mmod_face_detection_ex.cpp). Simplificado aqui.
template <template <typename> class BN, int N, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,1,1,SUBNET>>>>>;

// Exemplo de alguns níveis (você pode copiar a definição exata do repositório da dlib, se preferir)
template <typename SUBNET> using level0 = block<dlib::bn_con, 32, SUBNET>;
template <typename SUBNET> using level1 = block<dlib::bn_con, 32, level0<SUBNET>>;
template <typename SUBNET> using level2 = block<dlib::bn_con, 16, level1<SUBNET>>;
template <typename SUBNET> using level3 = block<dlib::bn_con, 8,  level2<SUBNET>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,
                           block<dlib::bn_con,32,
                           block<dlib::bn_con,32,
                           block<dlib::bn_con,16,
                           block<dlib::bn_con,8,
                           dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>
                           >>>>>>;
// ----------------------------------------------------------

int main(int argc, char* argv[]) {
    my::evm evm_processor;
    my::SignalProcessor signalProcessor;

    Database db;
    if (!db.open("/home/aldo/Documentos/TCC-EVM/server/measurement.db")) {
        std::cerr << "Falha ao abrir o banco de dados" << std::endl;
        return 1;
    }
    db.createTable();

    // Carregar o shape predictor (CPU)
    dlib::shape_predictor pose_model;
    dlib::deserialize("/home/aldo/Documentos/TCC-EVM/models/shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Carregar o detector de faces baseado em CNN (GPU)
    net_type net;
    dlib::deserialize("/home/aldo/Documentos/TCC-EVM/models/mmod_human_face_detector.dat") >> net;

    // Abrir captura de vídeo
    cv::VideoCapture cap(VIDEO_FILE_PATH);
    bool success = cap.isOpened();
    if (!success) {
        std::cerr << "Não foi possível abrir o vídeo." << std::endl;
        return 1;
    }

#if SHOW_FPS
    float sum = 0;
    int count = 0;
    float fps = 0;
#endif

    float lowFreq = 0.83f;  // Frequência cardíaca mínima (~50 bpm)
    float highFreq = 3.0f;  // Frequência cardíaca máxima (~180 bpm)
    float alpha    = 50.0f; // Fator de amplificação

    double heartRate = 0.0;
    double spo2      = 0.0;

    while (true) {
        cv::Mat frame;
        success = cap.read(frame);
        if (!success) break;

        // Espelha o frame horizontalmente (opcional)
        cv::flip(frame, frame, 1);

    #if SHOW_FPS
        auto start = std::chrono::high_resolution_clock::now();
    #endif

    dlib::cv_image<dlib::bgr_pixel> cimg(frame);
    dlib::matrix<dlib::rgb_pixel> dlibFrame;
    dlib::assign_image(dlibFrame, dlib::cv_image<dlib::bgr_pixel>(frame));
    std::vector<dlib::mmod_rect> faces = net(dlibFrame);



    #if SHOW_FPS
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float inferenceTime = duration.count() / 1e3;
        sum  += inferenceTime;
        count++;
        fps   = 1e3f / inferenceTime;
        cv::putText(frame,
                    std::to_string(static_cast<int>(fps)) + " FPS",
                    cv::Point(20, 70),
                    cv::FONT_HERSHEY_PLAIN,
                    2, cv::Scalar(0, 196, 255),
                    2);
    #endif

        // Loop pelas detecções
        for (auto& detectedFace : faces) {
            // Extraímos o retângulo (bounding box)
            dlib::rectangle faceRect = detectedFace.rect;

            // Encontra os landmarks (shape) para cada rosto detectado (CPU)
            dlib::full_object_detection shape = pose_model(cimg, faceRect);

            // Exemplo: calcular retângulos da testa e bochecha
            // (mesma lógica que antes, aproveitando shape.part(...))
            cv::Point foreheadTop(shape.part(19).x(), shape.part(19).y());
            cv::Point foreheadBottom(shape.part(24).x(), shape.part(24).y());
            cv::Rect foreheadROI(foreheadTop, foreheadBottom);

            cv::Point cheekLeft(shape.part(1).x(), shape.part(1).y());
            cv::Point cheekRight(shape.part(12).x(), shape.part(12).y());
            cv::Rect cheekROI(cheekLeft, cheekRight);

            bool useForehead  = true; 
            cv::Rect selectedRoi = useForehead ? foreheadROI : cheekROI;

            // Ajusta ROI caso exceda o tamanho do frame
            selectedRoi &= cv::Rect(0, 0, frame.cols, frame.rows);

            // Verificar se a ROI é válida
            if (selectedRoi.width > 1 && selectedRoi.height > 1) {
                cv::Mat selectedRegion = frame(selectedRoi);
                if (!selectedRegion.empty()) {
                    // Converte para YCrCb
                    cv::Mat ycrcb_selected;
                    cv::cvtColor(selectedRegion, ycrcb_selected, cv::COLOR_BGR2YCrCb);

                    // Separa os canais
                    std::vector<cv::Mat> ycrcb_channels;
                    cv::split(ycrcb_selected, ycrcb_channels);

                    // Checar canal Cb (índice 2)
                    if (ycrcb_channels.size() == 3 && !ycrcb_channels[2].empty()) {
                        // EVM no canal Cb
                        cv::Mat processed_channel = evm_processor.processChannel(
                            ycrcb_channels[2],
                            lowFreq, highFreq,
                            fps, alpha
                        );

                        if (!processed_channel.empty()) {
                            // Substitui o canal Cb e converte de volta
                            ycrcb_channels[2] = processed_channel;

                            cv::Mat processed_ycrcb;
                            cv::merge(ycrcb_channels, processed_ycrcb);

                            cv::Mat processed_bgr;
                            cv::cvtColor(processed_ycrcb, processed_bgr, cv::COLOR_YCrCb2BGR);

                            // Se o tamanho mudou, ajusta
                            if (processed_bgr.size() != selectedRegion.size()) {
                                cv::resize(processed_bgr, processed_bgr, selectedRegion.size());
                            }

                            // Copia a região processada de volta
                            processed_bgr.copyTo(frame(selectedRoi));

                            // Atualizar o processador de sinais
                            std::vector<cv::Mat> rgb_channels;
                            cv::split(processed_bgr, rgb_channels);

                            if (rgb_channels.size() == 3) {
                                signalProcessor.addFrameData(rgb_channels);
                            }
                        }
                    }
                }
            }

            // Desenhar o retângulo do rosto
            cv::Rect rectFace(
                faceRect.left(),
                faceRect.top(),
                faceRect.width(),
                faceRect.height()
            );

            // Como a ROI foi calculada em coords do shape, ajustamos:
            cv::Point topLeft(
                faceRect.left() + selectedRoi.x,
                faceRect.top()  + selectedRoi.y
            );
            cv::Point bottomRight(
                faceRect.left() + selectedRoi.x + selectedRoi.width,
                faceRect.top()  + selectedRoi.y + selectedRoi.height
            );

            cv::rectangle(frame, rectFace, cv::Scalar(0,255,0), 2);  // Rosto
            cv::rectangle(frame, cv::Rect(topLeft, bottomRight),
                          cv::Scalar(255, 0, 0), 2);                 // ROI
        }

        // Quando o buffer tiver 150 amostras, calcular HR e SpO2
        if (signalProcessor.getGreenChannelMeans().size() == 150) {

            auto t  = std::time(nullptr);
            auto tm = *std::localtime(&t);

            char buffer[100];
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm);
            std::string timestampStr(buffer);

            heartRate = signalProcessor.computeHeartRate(fps, timestampStr);
            spo2      = signalProcessor.computeSpO2();

            std::cout << "Estimated Heart Rate: " << heartRate << " bpm" << std::endl;
            std::cout << "Estimated SpO₂: "     << spo2      << "%"    << std::endl;

            // Salvar parâmetros intermediários
            signalProcessor.saveIntermediateParameters("/home/aldo/data/spo2_intermediate_params.csv");

            // Salvar frame
            std::string fileName = "/home/aldo/data/captures/" + timestampStr + ".png";
            cv::imwrite(fileName, frame);

            // Salvar no banco
            std::string relativePath = "captures/" + timestampStr + ".png";
            db.insertMeasurement(heartRate, spo2, relativePath);

            // Limpar buffer do SignalProcessor
            signalProcessor.reset();
        }

        // Mostrar resultados
        if (heartRate > 0.0) {
            cv::putText(frame,
                        "HR: " + std::to_string(static_cast<int>(heartRate)) + " bpm",
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(255, 0, 0),
                        2);
        }
        if (spo2 > 0.0) {
            cv::putText(frame,
                        "SpO2: " + std::to_string(static_cast<int>(spo2)) + " %",
                        cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(0, 0, 255),
                        2);
        }

        cv::imshow("Face Detector (CNN)", frame);
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
