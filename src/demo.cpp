#include "FaceDetection.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include "evm.hpp"
#include <opencv2/imgproc.hpp>

#define SHOW_FPS    (1)

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
    #endif

    while (success) {
        cv::Mat rframe;
        success = cap.read(rframe);
        if (!success) break;
        cv::flip(rframe, rframe, 1);

        #if SHOW_FPS
        auto start = std::chrono::high_resolution_clock::now();
        #endif

        faceDetector.loadImageToInput(rframe);
        faceDetector.runInference();

        #if SHOW_FPS
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float inferenceTime = duration.count() / 1e3;
        sum += inferenceTime;
        count += 1;
        int fps = (int)1e3 / inferenceTime;
        cv::putText(rframe, std::to_string(fps), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 196, 255), 2);
        #endif

        cv::Rect roi = faceDetector.getFaceRoi();
        if (roi.width > 0 && roi.height > 0) {
            cv::Mat croppedFace = faceDetector.cropFrame(roi);

            // Processamento diretamente no espaço de cor BGR (sem conversão)
            std::vector<cv::Mat> bgr_channels;
            cv::split(croppedFace, bgr_channels); 

            //cv::Mat b_channel = bgr_channels[0];
            //cv::Mat g_channel = bgr_channels[1];
           // cv::Mat r_channel = bgr_channels[2];
           // float lowFreq = 0.83, highFreq = 3, video_fps = 2000;
            float lowFreq = 20, highFreq = 40, video_fps = fps;
            std::cout << "Processando b" << std::endl;
            cv::Mat b_channel = evm_processor.processChannel(bgr_channels[0], lowFreq, highFreq, video_fps);
            std::cout << "Processando g" << std::endl;
            cv::Mat g_channel = evm_processor.processChannel(bgr_channels[1], lowFreq, highFreq, video_fps);
            std::cout << "Processando Cb" << std::endl;
            cv::Mat r_channel = evm_processor.processChannel(bgr_channels[2], lowFreq, highFreq, video_fps);

            // Normalização dos canais BGR
            cv::normalize(b_channel, b_channel, 0, 255, cv::NORM_MINMAX);
            cv::normalize(g_channel, g_channel, 0, 255, cv::NORM_MINMAX);
            cv::normalize(r_channel, r_channel, 0, 255, cv::NORM_MINMAX);

            // Convertendo para CV_8U para visualização
            b_channel.convertTo(b_channel, CV_8U);
            g_channel.convertTo(g_channel, CV_8U);
            r_channel.convertTo(r_channel, CV_8U);

            std::vector<cv::Mat> merged_bgr = {b_channel, g_channel, r_channel};
            cv::Mat processed_bgr;
            cv::merge(merged_bgr, processed_bgr); 

            // A imagem já está em BGR, então não é necessária a conversão de volta
            cv::imshow("Enhanced Face", processed_bgr);
        } else {
            cv::imshow("Face Detector", rframe);
        }


        if (cv::waitKey(10) == 27) break;  // Exit if 'ESC' is pressed
    }

    #if SHOW_FPS
    std::cout << "Average inference time: " << sum / count << "ms" << std::endl;
    #endif

    cap.release();
    cv::destroyAllWindows();
    return 0;
}


