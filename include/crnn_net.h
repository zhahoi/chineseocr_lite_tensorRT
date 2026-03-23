#ifndef __CRNN_NET_H__
#define __CRNN_NET_H__

#include "NvInferPlugin.h"
#include "common.hpp"
#include "ocr_struct.hpp"
#include "ocr_utils.h"
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class CrnnNet {

public:
    explicit CrnnNet(const std::string& engine_file_path);
    ~CrnnNet();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image, int dst_width);

    void infer();

    bool loadKeys(const std::string& keys_path);

    std::vector<TextLine> getTextLines(
        const std::vector<cv::Mat>& partImgs, 
        const char* path, 
        const char* imgName);

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
    static constexpr int DST_HEIGHT = 32;

    static constexpr int SEQ_LEN = 64;
    static constexpr int NUM_CLASSES = 5531;

    std::vector<std::string> keys;
    void set_input_shape(int dst_width);

    TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w);
    TextLine getTextLine(const cv::Mat &src);

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};

    bool isOutputDebugImg = false;
};



#endif // __CRNN_NET_H__