#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#pragma once

class Stereo {

public:
    enum class cost_function {
        rank
        , census
    };

    Stereo(int window_size, int max_dis, int tran_size,cost_function cost) {
        win_size_ = window_size;  // block size
        max_disparity_ = max_dis;
        tran_win_size_ = tran_size; // matching cost windows size
        cost_ = cost;
    }
    cv::Mat rank_transform(cv::Mat image, int tran_size);
    cv::Mat census_transform(cv::Mat image, int tran_size);
    cv::Mat stereo_match(cv::Mat left, cv::Mat right);

private:
    int win_size_, max_disparity_, tran_win_size_;
    cost_function cost_;
};