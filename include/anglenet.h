#ifndef __ANGLE_NET_H__
#define __ANGLE_NET_H__

#include "NvInferPlugin.h"
#include "common.hpp"
#include "ocr_struct.hpp"
#include "ocr_utils.h"
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class AngleNet {

public:
    explicit AngleNet(const std::string& engine_file_path);
    ~AngleNet();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void infer();

    std::vector<Angle> getAngles(
        const std::vector<cv::Mat>& partImgs, 
        const char* path, 
        const char* imgName, 
        bool doAngle, 
        bool mostAngle);

public:
    int                       num_bindings;
    int                       num_inputs  = 0;
    int                       num_outputs = 0;
    std::vector<Binding>      input_bindings;
    std::vector<Binding>      output_bindings;
    std::vector<void*>        host_ptrs;
    std::vector<void*>        device_ptrs;

private:
    static constexpr float MEAN_VALUES[3] = {127.5f, 127.5f, 127.5f};
    static constexpr float NORM_VALUES[3] = {1.0 / 127.5f, 1.0 / 127.5f, 1.0 / 127.5f};
    static constexpr int DST_WIDTH = 192;
    static constexpr int DST_HEIGHT = 32;

    Angle getAngle(cv::Mat &src);
    Angle postprocess_single();

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};

    bool isOutputAngleImg = false;
};



#endif // __ANGLE_NET_H__