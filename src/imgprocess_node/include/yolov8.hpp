#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
using namespace det;

inline bool Compition(const Object &a, const Object &b) {return a.prob > b.prob;};

const std::vector<std::string> CLASS_NAMES = {
    "ball", "bucket", "covered_ball"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189}};

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 detect(const cv::Mat& image);
    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs,
                                     std::vector<Object>& objs_bucket,
                                     std::vector<Object>& objs_ball,
                                     float                score_thres = 0.65f,
                                     float                iou_thres   = 0.65f,
                                     int                  topk        = 100,
                                     int                  num_labels  = CLASS_NAMES.size());
    static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      bool&                                         resultFlag,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    cv::Mat  res;
    cv::Size size        = cv::Size{640, 640};
    int      num_labels  = 1;
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.65f;
    std::vector<Object> objs;
    std::vector<Object> objs_bucket;
    std::vector<Object> objs_ball;
    bool     resultFlag  = false;


    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};


#endif  // DETECT_NORMAL_YOLOV8_HPP
