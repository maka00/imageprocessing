#include "ssd_match.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
static inline int HammingDistance(int a, int b) { return static_cast<int>( __builtin_popcount(a ^ b)); }

cv::Mat Stereo::rank_transform(cv::Mat image, int windowsize) {
    const int h = image.rows;
    const int w = image.cols;
    cv::Mat imgDisparity8U = cv::Mat(image.rows, image.cols, CV_8U);
    const int window_half = windowsize / 2;

    for (int y = window_half; y < h - window_half; ++y) {
        for (int x = window_half; x < w - window_half; ++x) {
            int ssd = 0;

            for (int v = -window_half; v < window_half + 1; ++v) {
                for (int u = -window_half; u < window_half + 1; ++u) {

                    if (image.at<uchar>(y + v, x + u) > image.at<uchar>(y, x)) ++ssd;
                }

            }

            imgDisparity8U.at<uchar>(y, x) = ssd;

        }
    }
    return imgDisparity8U;
}

cv::Mat Stereo::census_transform(cv::Mat image, int windowsize) {
    const int h = image.rows;
    const int w = image.cols;
    cv::Mat imgDisparity8U = cv::Mat(image.rows, image.cols, CV_8U);
    const int window_half = windowsize / 2;

    for (int y = window_half; y < h - window_half; ++y) {
        for (int x = window_half; x < w - window_half; ++x) {
            int ssd = 0;

            for (int v = -window_half; v < window_half + 1; ++v) {
                for (int u = -window_half; u < window_half + 1; ++u) {
                    if (v != 0 && u != 0) { // skip the central pixel
                        ssd <<= 1;
                        if (image.at<uchar>(y + v, x + u) > image.at<uchar>(y, x))
                            ssd = ssd +
                                  1; // assign last digit to 1 if pixel is larger than central pixel in the windows else assign 0
                    }
                }

            }

            imgDisparity8U.at<uchar>(y, x) = ssd;

        }
    }
    return imgDisparity8U;
}

cv::Mat Stereo::stereo_match(cv::Mat left, cv::Mat right) {
    const int h = left.rows - 1;
    const int w = left.cols - 1;
    cv::Mat imgDisparity8U = cv::Mat(left.rows, left.cols, CV_8UC1,cv::Scalar::all(0));
    const int window_half = win_size_ / 2;
    const int adjust = 255 / max_disparity_;
    //decide which matching cost function to use
    if (cost_ == cost_function::rank) {
        left = Stereo::rank_transform(left, tran_win_size_);
        right = Stereo::rank_transform(right, tran_win_size_);
    } else if (cost_ == cost_function::census) {
        left = Stereo::census_transform(left, tran_win_size_);
        right = Stereo::census_transform(right, tran_win_size_);
    }
    std::vector<int> rows(h - window_half);
    std::iota(std::begin(rows), std::end(rows), window_half);
    std::for_each(std::execution::par, std::begin(rows), std::end(rows),[&](const auto& y) {
    //for (int y = window_half; y < h - window_half; ++y) {
        uchar *imgDisparity_y = imgDisparity8U.ptr(y);
        for (int x = window_half; x < w - window_half; ++x) {
            int prev_ssd = INT_MAX;
            int best_dis = 0;
            for (int off = 0; off < max_disparity_; ++off) {
                int ssd = 0;
                int ssd_tmp = 0;
                for (int v = -window_half; v < window_half; ++v) {
                    for (int u = -window_half; u < window_half; ++u) {
                        if (cost_ == cost_function::census) {
                            auto i0 = std::clamp(y + v,0, h);
                            auto i1 = x + u;
                            uchar pix_left = left.at<uchar>(i0, i1);

                            auto i2 = std::abs(x + u - off);
                            uchar pix_right = right.at<uchar>(i0, i2);
                            ssd_tmp = HammingDistance(pix_left, pix_right);
                        } else {
                            ssd_tmp = left.at<uchar>(y + v, x + u) - right.at<uchar>(y + v, x + u - off);
                        }
                        ssd += ssd_tmp * ssd_tmp;
                    }
                }
                if (ssd < prev_ssd) {
                    prev_ssd = ssd;
                    best_dis = off;
                }
            }
            imgDisparity_y[x] = best_dis * adjust;
        }
    //}
    });
    return imgDisparity8U;
}
