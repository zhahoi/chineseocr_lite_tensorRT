#include "dbnet.h"
#include "ocr_utils.h"
#include <numeric>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>


DbNet::DbNet(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    initLibNvInferPlugins(&this->gLogger, "");

    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);

    delete[] trtModelStream;

    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    cudaStreamCreate(&this->stream);

    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        std::string name   = this->engine->getIOTensorName(i);
        binding.name       = name;
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
        binding.dsize      = type_to_size(dtype);

        bool isInput = this->engine->getTensorIOMode(name.c_str())
                       == nvinfer1::TensorIOMode::kINPUT;
        nvinfer1::Dims dims = this->engine->getProfileShape(
            name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

        if (isInput) {
            this->num_inputs += 1;
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            this->context->setInputShape(name.c_str(), dims);
        } else {
            dims             = this->context->getTensorShape(name.c_str());
            binding.size     = get_size_by_dims(dims);
            binding.dims     = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

DbNet::~DbNet()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;

    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void DbNet::make_pipe(bool warmup)
{
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void* d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; ++i) {
            for (auto& bindings : this->input_bindings) {
                size_t size = bindings.size * bindings.dsize;
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr,
                                      size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void DbNet::set_input_shape(int dst_width, int dst_height)
{
    const std::string& name = this->input_bindings[0].name;
    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0]   = 1;
    dims.d[1]   = 3;
    dims.d[2]   = dst_height;
    dims.d[3]   = dst_width;

    this->context->setInputShape(name.c_str(), dims);

    this->input_bindings[0].dims = dims;
    this->input_bindings[0].size = get_size_by_dims(dims);
}

void DbNet::copy_from_Mat(const cv::Mat& image, int dst_width, int dst_height)
{
    int channel_size = dst_width * dst_height;
    std::vector<float> data(3 * channel_size);

    for (int row = 0; row < dst_height; ++row) {
        const uchar* uc_pixel = image.data + row * image.step;
        for (int col = 0; col < dst_width; ++col) {
            int i = row * dst_width + col;
            data[i]                    = ((float)uc_pixel[2] - MEAN_VALUES[0]) * NORM_VALUES[0]; // R
            data[i + channel_size]     = ((float)uc_pixel[1] - MEAN_VALUES[1]) * NORM_VALUES[1]; // G
            data[i + 2 * channel_size] = ((float)uc_pixel[0] - MEAN_VALUES[2]) * NORM_VALUES[2]; // B
            uc_pixel += 3;
        }
    }

    size_t actual_size = 3 * channel_size * input_bindings[0].dsize;
    CHECK(cudaMemcpyAsync(device_ptrs[0], data.data(),
                          actual_size, cudaMemcpyHostToDevice, stream));
}

void DbNet::infer()
{
    for (int i = 0; i < this->num_bindings; ++i) {
        const std::string& name = (i < this->num_inputs)
            ? this->input_bindings[i].name
            : this->output_bindings[i - this->num_inputs].name;
        this->context->setTensorAddress(name.c_str(), this->device_ptrs[i]);
    }

    this->context->enqueueV3(this->stream);

    for (int i = 0; i < this->num_outputs; ++i) {
        const std::string& name = this->output_bindings[i].name;
        nvinfer1::Dims out_dims = this->context->getTensorShape(name.c_str());
        this->output_bindings[i].dims = out_dims;
        this->output_bindings[i].size = get_size_by_dims(out_dims);

        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i],
            this->device_ptrs[i + this->num_inputs],
            osize, cudaMemcpyDeviceToHost, this->stream));
    }

    CHECK(cudaStreamSynchronize(this->stream));
}

std::vector<TextBox> DbNet::findRsBoxes(
    const cv::Mat& fMapMat,
    const cv::Mat& norfMapMat,
    ScaleParam& s,
    float boxScoreThresh,
    float unClipRatio)
{
    float minArea = 3;
    std::vector<TextBox> rsBoxes;

    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < (int)contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;

        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;

        // Clipper 膨胀
        std::vector<cv::Point> clipBox    = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);

        if (minSideLen < minArea + 2)
            continue;

        // 坐标还原到原图，并 clamp
        for (int j = 0; j < (int)clipMinBox.size(); ++j) {
            clipMinBox[j].x = (clipMinBox[j].x / s.ratioWidth);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), s.srcWidth);
            clipMinBox[j].y = (clipMinBox[j].y / s.ratioHeight);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), s.srcHeight);
        }
        rsBoxes.emplace_back(TextBox{clipMinBox, score});
    }

    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

std::vector<TextBox> DbNet::getTextBoxes(
    cv::Mat& src,
    ScaleParam& s,
    float boxScoreThresh,
    float boxThresh,
    float unClipRatio)
{
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));

    set_input_shape(s.dstWidth, s.dstHeight);

    copy_from_Mat(srcResize, s.dstWidth, s.dstHeight);

    infer();

    nvinfer1::Dims out_dims = this->output_bindings[0].dims;
    int outH, outW;
    if (out_dims.nbDims == 4) {
        outH = out_dims.d[2];
        outW = out_dims.d[3];
    } else { // nbDims == 3
        outH = out_dims.d[1];
        outW = out_dims.d[2];
    }

    cv::Mat fMapMat(outH, outW, CV_32FC1);
    memcpy(fMapMat.data,
           static_cast<float*>(this->host_ptrs[0]),
           outH * outW * sizeof(float));

    cv::Mat norfMapMat = fMapMat > boxThresh;

    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}