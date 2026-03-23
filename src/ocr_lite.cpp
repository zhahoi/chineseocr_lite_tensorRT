#include "ocr_lite.h"
#include "ocr_utils.h"   
#include <stdarg.h>
#include <sstream>

OcrLite::OcrLite(
    const std::string& detPath,
    const std::string& clsPath,
    const std::string& recPath,
    const std::string& keysPath,
    const std::string& fontPath, 
    bool warmup)
    : dbNet(detPath),
      angleNet(clsPath),
      crnnNet(recPath)
{
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData(fontPath, 0);
    Logger("FreeType font loaded: %s\n", fontPath.c_str());

    Logger("=====Init Models=====\n");

    Logger("--- make_pipe DbNet ---\n");
    dbNet.make_pipe(warmup);

    Logger("--- make_pipe AngleNet ---\n");
    angleNet.make_pipe(warmup);

    Logger("--- make_pipe CrnnNet ---\n");
    crnnNet.make_pipe(warmup);

    Logger("--- Load CrnnNet Keys ---\n");
    if (!crnnNet.loadKeys(keysPath)) {
        fprintf(stderr, "[ERROR] Failed to load keys: %s\n", keysPath.c_str());
    }

    Logger("=====Init Models Done=====\n");
}

void OcrLite::Logger(const char* format, ...)
{
    if (!(isOutputConsole || isOutputResultTxt)) return;
    char* buffer = (char*)malloc(8192);
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    if (isOutputConsole)   printf("%s", buffer);
    if (isOutputResultTxt) fprintf(resultTxt, "%s", buffer);
    free(buffer);
}

void OcrLite::initLogger(bool isConsole, bool isPartImg, bool isResultImg)
{
    isOutputConsole   = isConsole;
    isOutputPartImg   = isPartImg;
    isOutputResultImg = isResultImg;
}

void OcrLite::enableResultTxt(const char* path, const char* imgName)
{
    isOutputResultTxt = true;
    std::string resultTxtPath = getResultTxtFilePath(path, imgName);
    printf("resultTxtPath(%s)\n", resultTxtPath.c_str());
    resultTxt = fopen(resultTxtPath.c_str(), "w");
}

static cv::Mat makePadding(cv::Mat& src, const int padding)
{
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc,
                       padding, padding, padding, padding,
                       cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}

OcrResult OcrLite::detect(
    const char* path, const char* imgName,
    int padding, int maxSideLen,
    float boxScoreThresh, float boxThresh,
    float unClipRatio, bool doAngle, bool mostAngle)
{
    std::string imgFile = getSrcImgFilePath(path, imgName);
    cv::Mat bgrSrc = cv::imread(imgFile, cv::IMREAD_COLOR); // BGR
    cv::Mat originSrc;
    cv::cvtColor(bgrSrc, originSrc, cv::COLOR_BGR2RGB);     // → RGB

    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize = (maxSideLen <= 0 || maxSideLen > originMaxSide)
                 ? originMaxSide : maxSideLen;
    resize += 2 * padding;

    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat  paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale    = getScaleParam(paddingSrc, resize);

    return detect(path, imgName, paddingSrc, paddingRect, scale,
                  boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
}

OcrResult OcrLite::detect(
    const cv::Mat& mat,
    int padding, int maxSideLen,
    float boxScoreThresh, float boxThresh,
    float unClipRatio, bool doAngle, bool mostAngle)
{
    cv::Mat originSrc;
    cv::cvtColor(mat, originSrc, cv::COLOR_BGR2RGB);        // → RGB

    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize = (maxSideLen <= 0 || maxSideLen > originMaxSide)
                 ? originMaxSide : maxSideLen;
    resize += 2 * padding;

    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat  paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale    = getScaleParam(paddingSrc, resize);

    return detect(nullptr, nullptr, paddingSrc, paddingRect, scale,
                  boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
}

std::vector<cv::Mat> OcrLite::getPartImages(
    cv::Mat& src,
    std::vector<TextBox>& textBoxes,
    const char* path,
    const char* imgName)
{
    std::vector<cv::Mat> partImages;
    for (int i = 0; i < (int)textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
        if (isOutputPartImg && path && imgName) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-part-");
            saveImg(partImg, debugImgFile.c_str());
        }
    }
    return partImages;
}

void OcrLite::drawTextResults(
    cv::Mat& img,
    const std::vector<TextBlock>& textBlocks,
    int thickness)
{
    if (textBlocks.empty()) return;

    const int panelWidth  = 300;
    const int panelHeight = img.rows;
    const int padding     = 8;
    const int lineSpacing = 4;

    cv::Mat panel(panelHeight, panelWidth, img.type(),
                  cv::Scalar(30, 30, 30)); 

    int fontSize = 20;
    int totalLines = (int)textBlocks.size();

    while (fontSize > 10) {
        cv::Size testSize = ft2->getTextSize("测", fontSize, -1, nullptr);
        int totalH = totalLines * (testSize.height + lineSpacing) + padding * 2;
        if (totalH <= panelHeight) break;
        fontSize -= 1;
    }

    cv::Size charSize = ft2->getTextSize("测", fontSize, -1, nullptr);
    int lineHeight = charSize.height + lineSpacing;

    int curY = padding + charSize.height;

    for (int i = 0; i < totalLines; ++i) {
        std::string label = std::to_string(i + 1) + ".";
        cv::Size labelSize = ft2->getTextSize(label, fontSize, -1, nullptr);

        ft2->putText(panel, label,
                     cv::Point(padding, curY),
                     fontSize,
                     cv::Scalar(160, 160, 160),  
                     -1, cv::LINE_AA, true);

        std::string text = textBlocks[i].text;
        int textX = padding + labelSize.width + 4;
        int maxTextW = panelWidth - textX - padding;

        cv::Size textSize = ft2->getTextSize(text, fontSize, -1, nullptr);
        while (textSize.width > maxTextW && text.size() > 3) {
            text = text.substr(0, text.size() - 3) ;
            textSize = ft2->getTextSize(text + "…", fontSize, -1, nullptr);
        }
        if (text != textBlocks[i].text) text += "…";

        ft2->putText(panel, text,
                     cv::Point(textX, curY),
                     fontSize,
                     cv::Scalar(0, 255, 0),  
                     -1, cv::LINE_AA, true);

        cv::line(panel,
                 cv::Point(padding, curY + lineSpacing),
                 cv::Point(panelWidth - padding, curY + lineSpacing),
                 cv::Scalar(60, 60, 60), 1);

        curY += lineHeight;
    }

    for (int i = 0; i < (int)textBlocks.size(); ++i) {
        const std::vector<cv::Point>& box = textBlocks[i].boxPoint;

        cv::line(img, box[0], box[1], cv::Scalar(0, 0, 255), thickness);
        cv::line(img, box[1], box[2], cv::Scalar(0, 0, 255), thickness);
        cv::line(img, box[2], box[3], cv::Scalar(0, 0, 255), thickness);
        cv::line(img, box[3], box[0], cv::Scalar(0, 0, 255), thickness);

        std::string idx = std::to_string(i + 1);
        int numFontSize = std::max(10, thickness * 5);
        cv::Size numSize = ft2->getTextSize(idx, numFontSize, -1, nullptr);

        cv::Point numOrg(box[0].x, box[0].y);
        numOrg.x = std::max(0, numOrg.x);
        numOrg.y = std::max(numSize.height, numOrg.y);

        cv::rectangle(img,
                      cv::Rect(numOrg.x, numOrg.y - numSize.height,
                               numSize.width + 2, numSize.height + 2),
                      cv::Scalar(255, 255, 255), -1);

        ft2->putText(img, idx,
                     cv::Point(numOrg.x + 1, numOrg.y),
                     numFontSize,
                     cv::Scalar(255, 0, 0),  
                     -1, cv::LINE_AA, true);
    }

    cv::Mat result;
    cv::hconcat(img, panel, result);
    img = result;
}

OcrResult OcrLite::detect(
    const char* path, const char* imgName,
    cv::Mat& src, cv::Rect& originRect, ScaleParam& scale,
    float boxScoreThresh, float boxThresh,
    float unClipRatio, bool doAngle, bool mostAngle)
{
    cv::Mat textBoxPaddingImg = src.clone();
    int thickness = getThickness(src);

    Logger("=====Start detect=====\n");
    Logger("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)\n",
           scale.srcWidth, scale.srcHeight,
           scale.dstWidth, scale.dstHeight,
           scale.ratioWidth, scale.ratioHeight);

    Logger("---------- step: dbNet getTextBoxes ----------\n");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes =
        dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    double endDbNetTime = getCurrentTime();
    double dbNetTime    = endDbNetTime - startTime;
    Logger("dbNetTime(%fms)\n", dbNetTime);

    for (int i = 0; i < (int)textBoxes.size(); ++i) {
        Logger("TextBox[%d](+padding)[score(%f),"
               "[x:%d,y:%d],[x:%d,y:%d],[x:%d,y:%d],[x:%d,y:%d]]\n", i,
               textBoxes[i].score,
               textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
               textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
               textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
               textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
    }

    Logger("---------- step: drawTextBoxes ----------\n");
    drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    std::vector<cv::Mat> partImages =
        getPartImages(src, textBoxes, path, imgName);

    Logger("---------- step: angleNet getAngles ----------\n");
    std::vector<Angle> angles =
        angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);

    for (int i = 0; i < (int)angles.size(); ++i) {
        Logger("angle[%d][index(%d),score(%f),time(%fms)]\n",
               i, angles[i].index, angles[i].score, angles[i].time);
    }

    for (int i = 0; i < (int)partImages.size(); ++i) {
        if (angles[i].index == 0) {
            partImages.at(i) = matRotateClockWise180(partImages[i]);
        }
    }

    Logger("---------- step: crnnNet getTextLine ----------\n");
    std::vector<TextLine> textLines =
        crnnNet.getTextLines(partImages, path, imgName);

    for (int i = 0; i < (int)textLines.size(); ++i) {
        Logger("textLine[%d](%s)\n", i, textLines[i].text.c_str());
        std::ostringstream txtScores;
        for (int s = 0; s < (int)textLines[i].charScores.size(); ++s) {
            if (s == 0) txtScores << textLines[i].charScores[s];
            else        txtScores << " ," << textLines[i].charScores[s];
        }
        Logger("textScores[%d]{%s}\n", i, txtScores.str().c_str());
        Logger("crnnTime[%d](%fms)\n", i, textLines[i].time);
    }

    std::vector<TextBlock> textBlocks;
    for (int i = 0; i < (int)textLines.size(); ++i) {
        std::vector<cv::Point> boxPoint(4);
        int pad = originRect.x; // padding version
        for (int j = 0; j < 4; ++j) {
            boxPoint[j] = cv::Point(
                textBoxes[i].boxPoint[j].x - pad,
                textBoxes[i].boxPoint[j].y - pad);
        }
        TextBlock textBlock{
            boxPoint,
            textBoxes[i].score,
            angles[i].index,
            angles[i].score,
            angles[i].time,
            textLines[i].text,
            textLines[i].charScores,
            textLines[i].time,
            angles[i].time + textLines[i].time
        };
        textBlocks.emplace_back(textBlock);
    }

    double endTime  = getCurrentTime();
    double fullTime = endTime - startTime;
    Logger("=====End detect=====\n");
    Logger("FullDetectTime(%fms)\n", fullTime);

    // cropped to original size
    cv::Mat rgbBoxImg, textBoxImg;
    if (originRect.x > 0 && originRect.y > 0) {
        textBoxPaddingImg(originRect).copyTo(rgbBoxImg);
    } else {
        rgbBoxImg = textBoxPaddingImg;
    }
    cv::cvtColor(rgbBoxImg, textBoxImg, cv::COLOR_RGB2BGR);

    if (isOutputResultImg && path && imgName) {
        std::string resultImgFile = getResultImgFilePath(path, imgName);
        cv::imwrite(resultImgFile, textBoxImg);
    }

    std::string strRes;
    for (int i = 0; i < (int)textBlocks.size(); ++i) {
        strRes.append(textBlocks[i].text);
        strRes.append("\n");
    }

    drawTextResults(textBoxImg, textBlocks, thickness);

    return OcrResult{dbNetTime, textBlocks, textBoxImg, fullTime, strRes};
}