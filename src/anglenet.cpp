#include "anglenet.h"
#include <numeric>
#include <cassert>
#include <iostream>


AngleNet::AngleNet(const std::string& engine_file_path)
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

AngleNet::~AngleNet()
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

void AngleNet::make_pipe(bool warmup)
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

void AngleNet::copy_from_Mat(const cv::Mat& image)
{
    int channel_size = DST_WIDTH * DST_HEIGHT;
    std::vector<float> data(3 * channel_size);

    for (int row = 0; row < DST_HEIGHT; ++row) {
        const uchar* uc_pixel = image.data + row * image.step; 
        for (int col = 0; col < DST_WIDTH; ++col) {
            int i = row * DST_WIDTH + col;
            data[i]                    = ((float)uc_pixel[2] - MEAN_VALUES[0]) * NORM_VALUES[0]; // R
            data[i + channel_size]     = ((float)uc_pixel[1] - MEAN_VALUES[1]) * NORM_VALUES[1]; // G
            data[i + 2 * channel_size] = ((float)uc_pixel[0] - MEAN_VALUES[2]) * NORM_VALUES[2]; // B
            uc_pixel += 3;
        }
    }

    size_t size = input_bindings[0].size * input_bindings[0].dsize;
    CHECK(cudaMemcpyAsync(
        device_ptrs[0], data.data(),
        size, cudaMemcpyHostToDevice, stream));
}

void AngleNet::infer()
{
    for (int i = 0; i < this->num_bindings; ++i) {
        const std::string& name = (i < this->num_inputs)
            ? this->input_bindings[i].name
            : this->output_bindings[i - this->num_inputs].name;
        this->context->setTensorAddress(name.c_str(), this->device_ptrs[i]);
    }

    this->context->enqueueV3(this->stream);

    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size
                     * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i],
            this->device_ptrs[i + this->num_inputs],
            osize,
            cudaMemcpyDeviceToHost,
            this->stream));
    }

    CHECK(cudaStreamSynchronize(this->stream));
}

Angle AngleNet::postprocess_single()
{
    float* outputData = static_cast<float*>(this->host_ptrs[0]);
    int    len        = static_cast<int>(this->output_bindings[0].size); 

    int   maxIndex = 0;
    float maxScore = outputData[0];
    for (int i = 1; i < len; ++i) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    return Angle{maxIndex, maxScore};
}

Angle AngleNet::getAngle(cv::Mat& src)
{
    copy_from_Mat(src);   
    infer();            
    return postprocess_single(); 
}

std::vector<Angle> AngleNet::getAngles(
    const std::vector<cv::Mat>& partImgs,
    const char* path,
    const char* imgName,
    bool doAngle,
    bool mostAngle)
{
    int size = static_cast<int>(partImgs.size());
    std::vector<Angle> angles(size);

    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();

            auto angleImg = adjustTargetImg(
                const_cast<cv::Mat&>(partImgs[i]), DST_WIDTH, DST_HEIGHT);

            Angle angle = getAngle(angleImg);

            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;
            angles[i]  = angle;

            if (isOutputAngleImg) {
                std::string angleImgFile =
                    getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }

    if (doAngle && mostAngle) {
        auto   angleIndexes = getAngleIndexes(angles);
        double sum          = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent  = angles.size() / 2.0f;

        int mostAngleIndex = (sum < halfPercent) ? 0 : 1;
        printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);

        for (int i = 0; i < size; ++i) {
            angles[i].index = mostAngleIndex;
        }
    }

    return angles;
}