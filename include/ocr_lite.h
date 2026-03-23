#ifndef __OCR_LITE_H__
#define __OCR_LITE_H__

#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include "ocr_struct.hpp"
#include "dbnet.h"
#include "crnn_net.h"
#include "anglenet.h"

class OcrLite {
public:
    OcrLite(
        const std::string& detPath,
        const std::string& clsPath,
        const std::string& recPath,
        const std::string& keysPath,
        const std::string& fontPath,
        bool warmup = true);
    ~OcrLite() = default;

    void initLogger(bool isConsole, bool isPartImg, bool isResultImg);
    void enableResultTxt(const char* path, const char* imgName);
    void Logger(const char* format, ...);

    OcrResult detect(
        const char* path, const char* imgName,
        int padding, int maxSideLen,
        float boxScoreThresh, float boxThresh,
        float unClipRatio, bool doAngle, bool mostAngle);

    OcrResult detect(
        const cv::Mat& mat,
        int padding, int maxSideLen,
        float boxScoreThresh, float boxThresh,
        float unClipRatio, bool doAngle, bool mostAngle);

private:
    DbNet    dbNet;
    AngleNet angleNet;
    CrnnNet  crnnNet;
    cv::Ptr<cv::freetype::FreeType2> ft2;

    bool isOutputConsole   = false;
    bool isOutputPartImg   = false;
    bool isOutputResultTxt = false;
    bool isOutputResultImg = false;
    FILE* resultTxt        = nullptr;

    std::vector<cv::Mat> getPartImages(
        cv::Mat& src,
        std::vector<TextBox>& textBoxes,
        const char* path,
        const char* imgName);

    void drawTextResults(
        cv::Mat& img,
        const std::vector<TextBlock>& textBlocks,
        int thickness);

    OcrResult detect(
        const char* path, const char* imgName,
        cv::Mat& src, cv::Rect& originRect, ScaleParam& scale,
        float boxScoreThresh = 0.6f,
        float boxThresh      = 0.3f,
        float unClipRatio    = 2.0f,
        bool doAngle         = true,
        bool mostAngle       = true);
};

#endif // __OCR_LITE_H__
