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
// #include <dlib/image_processing/frontal_face_detector.h>
using namespace dlib;

#define SHOW_FPS (1)

#if SHOW_FPS
#include <chrono>
#endif

// Caminho do vídeo
const std::string VIDEO_FILE_PATH = "/home/aldo/Documentos/video/build/luz_natural_video_5s.avi";

// ----------------------------------------------------------
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
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
    dlib::deserialize("/home/aldo/Documentos/TCC-EVM/models/shape_predictor_5_face_landmarks.dat") >> pose_model;

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

    int frameCounter = 0;
    dlib::full_object_detection lastShape;

    while (true) {
        cv::Mat frame;
        success = cap.read(frame);
        if (!success) break;

        // Espelha o frame horizontalmente (opcional)
        cv::flip(frame, frame, 1);
        frameCounter++;

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
            dlib::full_object_detection shape;

            if (frameCounter % 150 == 0){
                shape = pose_model(cimg, faceRect);
                lastShape = shape;
            }
            else{
                if (lastShape.num_parts() == 0){
                    shape = pose_model(cimg, faceRect);
                    lastShape = shape;
                } else{
                    shape = lastShape;
                }

            }

            // Verifica se temos 5 pontos de fato
            if (shape.num_parts() == 5)
            {
                // 0: olho E, 1: olho D, 2: nariz, 3: boca E, 4: boca D
                cv::Point leftEye(
                    shape.part(0).x(),
                    shape.part(0).y()
                );
                cv::Point rightEye(
                    shape.part(1).x(),
                    shape.part(1).y()
                );
                cv::Point nose(
                    shape.part(2).x(),
                    shape.part(2).y()
                );

                // Desenha linha entre olho E e olho D
                cv::line(frame, leftEye, rightEye, cv::Scalar(0,255,255), 2);

                // Ponto médio entre os olhos
                cv::Point eyeMid(
                    (leftEye.x + rightEye.x)/2,
                    (leftEye.y + rightEye.y)/2
                );

                // Distância do centro dos olhos até o nariz (em Y ou euclidiana)
                // Se quiser a distância euclidiana:
                float distEuclid = cv::norm(cv::Point2f(eyeMid) - cv::Point2f(nose));

                // Vamos criar a ROI da "testa" pegando:
                // - Largura = distancia horizontal entre os olhos
                // - Altura  = distEuclid (mesmo valor da linha olho-meio até o nariz)
                // - Posição  = acima dos olhos (inverte o vetor)

                int eyeWidth = static_cast<int>(
                    std::round(cv::norm(cv::Point2f(rightEye) - cv::Point2f(leftEye)))
                );

                // Se a pessoa estiver normal, o nariz estará abaixo do eyeMid,
                // logo "subir" distEuclid em Y:
                // top = eyeMid.y - distEuclid
                // bottom = eyeMid.y
                // centerX ~ eyeMid.x - (eyeWidth/2)

                int rectWidth  = static_cast<int>(std::round(distEuclid));
                int rectHeight = eyeWidth;

                // Podemos alinhar o retângulo no eixo horizontal dos olhos
                int rectLeft   = eyeMid.x - (rectWidth/2);
                int rectTop    = eyeMid.y - rectHeight;
                int rectRight  = rectLeft + rectWidth;
                int rectBottom = eyeMid.y;

                cv::Rect foreheadROI(
                    cv::Point(rectLeft,  rectTop),
                    cv::Point(rectRight, rectBottom)
                );

                // Ajusta ROI para não sair da imagem
                foreheadROI &= cv::Rect(0, 0, frame.cols, frame.rows);

                if (foreheadROI.width > 1 && foreheadROI.height > 1)
                {
                    cv::Mat selectedRegion = frame(foreheadROI);

                    // Converte para YCrCb
                    cv::Mat ycrcb_selected;
                    cv::cvtColor(selectedRegion, ycrcb_selected, cv::COLOR_BGR2YCrCb);

                    // Separa os canais
                    std::vector<cv::Mat> ycrcb_channels;
                    cv::split(ycrcb_selected, ycrcb_channels);

                    // Canal Cb = index 2
                    if (ycrcb_channels.size() == 3 && !ycrcb_channels[2].empty())
                    {
                        // EVM no canal Cb
                        cv::Mat processed_channel = evm_processor.processChannel(
                            ycrcb_channels[2],
                            lowFreq, highFreq,
                            fps, alpha
                        );

                        if (!processed_channel.empty())
                        {
                            ycrcb_channels[2] = processed_channel;
                            cv::Mat processed_ycrcb;
                            cv::merge(ycrcb_channels, processed_ycrcb);

                            cv::Mat processed_bgr;
                            cv::cvtColor(processed_ycrcb, processed_bgr, cv::COLOR_YCrCb2BGR);

                            // Ajusta tamanho se mudou
                            if (processed_bgr.size() != selectedRegion.size()) {
                                cv::resize(processed_bgr, processed_bgr, selectedRegion.size());
                            }

                            // Copia de volta
                            processed_bgr.copyTo(frame(foreheadROI));

                            // Atualiza signalProcessor
                            std::vector<cv::Mat> rgb_channels;
                            cv::split(processed_bgr, rgb_channels);
                            if (rgb_channels.size() == 3) {
                                signalProcessor.addFrameData(rgb_channels);
                            }
                        }
                    }
                }

                // Desenha o retângulo da testa
                cv::rectangle(frame, foreheadROI, cv::Scalar(255,0,0), 2);

                // Desenha também bounding box do rosto (opcional)
                cv::Rect faceRectCV(
                    faceRect.left(),
                    faceRect.top(),
                    faceRect.width(),
                    faceRect.height()
                );
                cv::rectangle(frame, faceRectCV, cv::Scalar(0,255,0), 2);
            }

        } // Fim loop faces

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
