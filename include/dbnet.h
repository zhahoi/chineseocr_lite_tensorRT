#ifndef __DB_NET_H__
#define __DB_NET_H__

#include "NvInferPlugin.h"
#include "common.hpp"
#include "ocr_struct.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class DbNet {

public:
    explicit DbNet(const std::string& engine_file_path);
    ~DbNet();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image, int dst_width, int dst_height);

    void infer();

    std::vector<TextBox> getTextBoxes(
        cv::Mat &src, 
        ScaleParam &s, 
        float boxScoreThresh,                      
        float boxThresh, 
        float unClipRatio);

public:
    int                       num_bindings;
    int                       num_inputs  = 0;
    int                       num_outputs = 0;
    std::vector<Binding>      input_bindings;
    std::vector<Binding>      output_bindings;
    std::vector<void*>        host_ptrs;
    std::vector<void*>        device_ptrs;

private:
    static constexpr float MEAN_VALUES[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    static constexpr float NORM_VALUES[3] = {1.0f / (0.229f * 255.f), 1.0f / (0.229f * 255.f), 1.0f / (0.229f * 255.f)};

    void set_input_shape(int dst_width, int dst_height);

    static std::vector<TextBox> findRsBoxes(
        const cv::Mat& fMapMat,
        const cv::Mat& norfMapMat,
        ScaleParam& s,
        float boxScoreThresh,
        float unClipRatio);

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};



#endif // __DB_NET_H__