#include <cstdio>
#include <string>
#include <getopt.h>
#include "ocr_lite.h"
#include "ocr_utils.h"

static const option long_options[] = {
    {"models_dir",       required_argument, nullptr, 'd'},
    {"det",              required_argument, nullptr, '1'},
    {"cls",              required_argument, nullptr, '2'},
    {"rec",              required_argument, nullptr, '3'},
    {"keys",             required_argument, nullptr, '4'},
    {"image",            required_argument, nullptr, 'i'},
    {"padding",          required_argument, nullptr, 'p'},
    {"max_side_len",     required_argument, nullptr, 's'},
    {"box_score_thresh", required_argument, nullptr, 'b'},
    {"box_thresh",       required_argument, nullptr, 'o'},
    {"unclip_ratio",     required_argument, nullptr, 'u'},
    {"do_angle",         required_argument, nullptr, 'a'},
    {"most_angle",       required_argument, nullptr, 'A'},
    {"help",             no_argument,       nullptr, 'h'},
    {nullptr,            0,                 nullptr,  0 }
};

static void printHelp(FILE* out, const char* argv0)
{
    fprintf(out, "Usage: %s [options]\n", argv0);
    fprintf(out, "  -d  --models_dir        模型目录\n");
    fprintf(out, "  -1  --det               dbnet  引擎文件名 (默认: dbnet.engine)\n");
    fprintf(out, "  -2  --cls               anglenet 引擎文件名 (默认: anglenet.engine)\n");
    fprintf(out, "  -3  --rec               crnnnet 引擎文件名 (默认: crnnnet.engine)\n");
    fprintf(out, "  -4  --keys              keys 文件名 (默认: keys.txt)\n");
    fprintf(out, "  -i  --image             输入图片路径\n");
    fprintf(out, "  -p  --padding           padding 大小 (默认: 50)\n");
    fprintf(out, "  -s  --max_side_len      最长边限制 (默认: 1024)\n");
    fprintf(out, "  -b  --box_score_thresh  文本框置信度阈值 (默认: 0.6)\n");
    fprintf(out, "  -o  --box_thresh        二值化阈值 (默认: 0.3)\n");
    fprintf(out, "  -u  --unclip_ratio      膨胀比例 (默认: 2.0)\n");
    fprintf(out, "  -a  --do_angle          是否做方向检测 0/1 (默认: 1)\n");
    fprintf(out, "  -A  --most_angle        是否投票统一方向 0/1 (默认: 1)\n");
    fprintf(out, "  -h  --help              打印帮助\n");
}

int main(int argc, char** argv)
{
    if (argc <= 1) {
        printHelp(stderr, argv[0]);
        return -1;
    }

    std::string modelsDir;
    std::string modelDetName  = "dbnet.engine";
    std::string modelClsName  = "angle_net.engine";
    std::string modelRecName  = "crnn_lite_lstm.engine";
    std::string keysName      = "keys.txt";
    std::string fontName      = "NotoSansCJK-Regular.otf";
    std::string imgPath;

    int   padding        = 50;
    int   maxSideLen     = 1024;
    float boxScoreThresh = 0.6f;
    float boxThresh      = 0.3f;
    float unClipRatio    = 2.0f;
    bool  doAngle        = true;
    bool  mostAngle      = true;

    int opt, optionIndex = 0;
    while ((opt = getopt_long(argc, argv,
                              "d:1:2:3:4:i:p:s:b:o:u:a:A:h",
                              long_options, &optionIndex)) != -1) {
        switch (opt) {
            case 'd':
                modelsDir = optarg;
                printf("modelsDir=%s\n", modelsDir.c_str());
                break;
            case '1':
                modelDetName = optarg;
                printf("det engine=%s\n", modelDetName.c_str());
                break;
            case '2':
                modelClsName = optarg;
                printf("cls engine=%s\n", modelClsName.c_str());
                break;
            case '3':
                modelRecName = optarg;
                printf("rec engine=%s\n", modelRecName.c_str());
                break;
            case '4':
                keysName = optarg;
                printf("keys=%s\n", keysName.c_str());
                break;
            case 'f':
                fontName = optarg;
                printf("fontName=%s\n", fontName.c_str());
                break;
            case 'i':
                imgPath = optarg;
                printf("imgPath=%s\n", imgPath.c_str());
                break;
            case 'p':
                padding = (int)strtol(optarg, nullptr, 10);
                break;
            case 's':
                maxSideLen = (int)strtol(optarg, nullptr, 10);
                break;
            case 'b':
                boxScoreThresh = strtof(optarg, nullptr);
                break;
            case 'o':
                boxThresh = strtof(optarg, nullptr);
                break;
            case 'u':
                unClipRatio = strtof(optarg, nullptr);
                break;
            case 'a':
                doAngle = (strtol(optarg, nullptr, 10) != 0);
                break;
            case 'A':
                mostAngle = (strtol(optarg, nullptr, 10) != 0);
                break;
            case 'h':
                printHelp(stdout, argv[0]);
                return 0;
            default:
                printf("unknown option -%c\n", opt);
                break;
        }
    }

    std::string modelDetPath = modelsDir + "/" + modelDetName;
    std::string modelClsPath = modelsDir + "/" + modelClsName;
    std::string modelRecPath = modelsDir + "/" + modelRecName;
    std::string keysPath     = modelsDir + "/" + keysName;
    std::string fontPath     = modelsDir + "/" + fontName;

    if (!isFileExists(imgPath)) {
        fprintf(stderr, "Image not found: %s\n", imgPath.c_str());
        return -1;
    }
    if (!isFileExists(modelDetPath)) {
        fprintf(stderr, "DbNet engine not found: %s\n", modelDetPath.c_str());
        return -1;
    }
    if (!isFileExists(modelClsPath)) {
        fprintf(stderr, "AngleNet engine not found: %s\n", modelClsPath.c_str());
        return -1;
    }
    if (!isFileExists(modelRecPath)) {
        fprintf(stderr, "CrnnNet engine not found: %s\n", modelRecPath.c_str());
        return -1;
    }
     if (!isFileExists(fontPath)) {
        fprintf(stderr, "Font file not found: %s\n", fontPath.c_str());
        return -1;
    }
    if (!isFileExists(keysPath)) {
        fprintf(stderr, "Keys file not found: %s\n", keysPath.c_str());
        return -1;
    }

    std::string imgDir  = imgPath.substr(0, imgPath.find_last_of('/') + 1);
    std::string imgName = imgPath.substr(imgPath.find_last_of('/') + 1);
    printf("imgDir=%s, imgName=%s\n", imgDir.c_str(), imgName.c_str());

    OcrLite ocrLite(modelDetPath, modelClsPath, modelRecPath, keysPath, fontPath, /*warmup=*/true);

    ocrLite.initLogger(
        true,   // isOutputConsole
        false,  // isOutputPartImg
        true);  // isOutputResultImg

    ocrLite.enableResultTxt(imgDir.c_str(), imgName.c_str());

    ocrLite.Logger("=====Input Params=====\n");
    ocrLite.Logger(
        "padding(%d),maxSideLen(%d),boxScoreThresh(%f),"
        "boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d)\n",
        padding, maxSideLen, boxScoreThresh,
        boxThresh, unClipRatio, doAngle, mostAngle);

    OcrResult result = ocrLite.detect(
        imgDir.c_str(), imgName.c_str(),
        padding, maxSideLen,
        boxScoreThresh, boxThresh, unClipRatio,
        doAngle, mostAngle);

    ocrLite.Logger("%s\n", result.strRes.c_str());

    if (!result.boxImg.empty()) {
        std::string resultImgPath = imgDir + imgName + "-result.jpg";
        cv::imwrite(resultImgPath, result.boxImg);
        printf("Result image saved: %s\n", resultImgPath.c_str());
    }
    return 0;
}