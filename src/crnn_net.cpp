#include "crnn_net.h"
#include "ocr_utils.h"
#include <numeric>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>


CrnnNet::CrnnNet(const std::string& engine_file_path)
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

CrnnNet::~CrnnNet()
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

void CrnnNet::make_pipe(bool warmup)
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

bool CrnnNet::loadKeys(const std::string& keys_path)
{
    std::ifstream in(keys_path.c_str());
    if (!in) {
        printf("The keys.txt file was not found\n");
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        keys.push_back(line);
    }
    if (keys.size() != 5531) {
        fprintf(stderr, "missing keys\n");
        return false;
    }
    printf("total keys size(%lu)\n", keys.size());
    return true;
}

void CrnnNet::set_input_shape(int dst_width)
{
    const std::string& name = this->input_bindings[0].name;
    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0]   = 1;
    dims.d[1]   = 3;
    dims.d[2]   = DST_HEIGHT;
    dims.d[3]   = dst_width;

    this->context->setInputShape(name.c_str(), dims);

    this->input_bindings[0].dims = dims;
    this->input_bindings[0].size = get_size_by_dims(dims);
}

void CrnnNet::copy_from_Mat(const cv::Mat& image, int dst_width)
{
    int channel_size = dst_width * DST_HEIGHT;
    std::vector<float> data(3 * channel_size);

    for (int row = 0; row < DST_HEIGHT; ++row) {
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

void CrnnNet::infer()
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

TextLine CrnnNet::scoreToTextLine(const std::vector<float>& outputData, int h, int w)
{
    int keySize = static_cast<int>(keys.size());
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = 0;

    for (int i = 0; i < h; ++i) {
        std::vector<float> exps(w);
        for (int j = 0; j < w; ++j) {
            exps[j] = std::exp(outputData[i * w + j]);
        }
        float partition = std::accumulate(exps.begin(), exps.end(), 0.0f);

        int   maxIndex = static_cast<int>(
            std::distance(exps.begin(),
                          std::max_element(exps.begin(), exps.end())));
        float maxValue = exps[maxIndex] / partition;

        if (maxIndex > 0 && maxIndex < keySize
            && !(i > 0 && maxIndex == lastIndex)) {
            scores.push_back(maxValue);
            strRes.append(keys[maxIndex - 1]);
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores};
}

TextLine CrnnNet::getTextLine(const cv::Mat& src)
{
    float scale    = (float)DST_HEIGHT / (float)src.rows;
    int   dstWidth = static_cast<int>((float)src.cols * scale);

    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(dstWidth, DST_HEIGHT));

    set_input_shape(dstWidth);

    copy_from_Mat(srcResize, dstWidth);

    infer();

    nvinfer1::Dims out_dims = this->output_bindings[0].dims;

    int seq_len, num_classes;
    if (out_dims.nbDims == 3) {
        // [seq_len, batch=1, num_classes]
        seq_len     = out_dims.d[0];  
        num_classes = out_dims.d[2];  
    } else {
        // [seq_len, num_classes]
        seq_len     = out_dims.d[0];
        num_classes = out_dims.d[1];
    }

    float* floatArray = static_cast<float*>(this->host_ptrs[0]);
    std::vector<float> outputData(floatArray,
                                  floatArray + seq_len * num_classes);

    return scoreToTextLine(outputData, seq_len, num_classes);
}

std::vector<TextLine> CrnnNet::getTextLines(
    const std::vector<cv::Mat>& partImgs,
    const char* path,
    const char* imgName)
{
    int size = static_cast<int>(partImgs.size());
    std::vector<TextLine> textLines(size);

    for (int i = 0; i < size; ++i) {
        if (isOutputDebugImg) {
            std::string debugImgFile =
                getDebugImgFilePath(path, imgName, i, "-debug-");
            cv::Mat tmp = partImgs[i];
            saveImg(tmp, debugImgFile.c_str());
        }

        double startTime = getCurrentTime();
        TextLine textLine = getTextLine(partImgs[i]);
        double endTime   = getCurrentTime();

        textLine.time = endTime - startTime;
        textLines[i]  = textLine;
    }
    return textLines;
}