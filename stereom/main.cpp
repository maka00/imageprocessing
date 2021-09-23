#include <iostream>
#include "ssd_match.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <array>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

cv::Mat scale_image(const cv::Mat& image) {
    cv::Mat scaled_image;
    cv::resize(image, scaled_image,cv::Size(450, 375),0,0, cv::INTER_LINEAR );
    return scaled_image;
}


cv::Mat custom_stereomatching(cv::Mat &left, cv::Mat &right) {
    cv::Mat result;
    const int window_size = 6;
    const int max_disparity = 50;
    const int tranwin_size = 7;
    const Stereo::cost_function cost = Stereo::cost_function::census;
    Stereo s(window_size, max_disparity, tranwin_size, cost);
    SPDLOG_INFO("stereo matching performed.");

    result = s.stereo_match(left, right);
    return result;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr generate_pointcloud(cv::Mat &dis) {
    std::array<std::array<int,3>, 8> color_map_jet{{
                                                           {0x60, 0x48, 0x60},
                                                           {0x78, 0x48, 0x60},
                                                           {0xa8, 0x60, 0x60},
                                                           {0xc0, 0x78, 0x60},
                                                           {0xf0, 0xa8, 0x48},
                                                           {0xf8, 0xca, 0x8c},
                                                           {0xfe, 0xec, 0xae},
                                                           {0xff, 0xf4, 0xc2}
                                                   }};
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = std::make_unique<pcl::PointCloud<pcl::PointXYZRGB>>(pcl::PointCloud<pcl::PointXYZRGB>());
    for(int i = 0; i < dis.rows; i++) {
        for (int j = 0; j < dis.cols; j++) {
            int pt = dis.at<uchar>(i, j);
            uint8_t grey = static_cast<uint8_t>(pt);
            int color_entry = grey / 16;
            pcl::PointXYZRGB point = pcl::PointXYZRGB(i,j,pt, color_map_jet[color_entry][0],color_map_jet[color_entry][1],color_map_jet[color_entry][2]);
            cloud->push_back(point);
       }
    }
    return cloud;
}

cv::Mat openCV_stereomatching(cv::Mat& left, cv::Mat& right) {
    cv::Mat result;
    const int num_disparities = 0;
    const int block_size = 9;
    cv::Ptr<cv::StereoBM> l_matcher = cv::StereoBM::create(num_disparities,block_size);
    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(l_matcher);
    cv::Ptr<cv::StereoMatcher> r_matcher = cv::ximgproc::createRightMatcher(l_matcher);
    cv::Mat left_for_matcher = left.clone();
    cv::Mat right_for_matcher = right.clone();
    cv::cvtColor(left_for_matcher,left_for_matcher,cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_for_matcher,right_for_matcher,cv::COLOR_BGR2GRAY);
    SPDLOG_INFO("stereo matching convert to greyscale.");

    cv::Mat left_disp, right_disp;
    l_matcher->compute(left_for_matcher,right_for_matcher,left_disp);
    l_matcher->compute(right_for_matcher,left_for_matcher,right_disp);
    SPDLOG_INFO("stereo matching calculate disparity.");

    const double lambda = 8000.0;
    const double sigma = 1.5;
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    cv::Mat filtered_disp;
    wls_filter->filter(left_disp, left, filtered_disp, right_disp);
    SPDLOG_INFO("stereo matching apply filter to disparity.");
    cv::Mat visual_disp;
    cv::ximgproc::getDisparityVis(filtered_disp, visual_disp);
    result = visual_disp;
    return result;
}

int main(int argc, char** argv) {
    static auto console = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("[%H:%M:%S.%e][%s][%!][%#] %v");
    const cv::String keys =
            "{help h usage ? | | usage...}"
            "{left  |<none> | left image}"
            "{right |<none> | left image}"
            "{out   |a_out   | depth map}"
            ;
    cv::CommandLineParser parser(argc, argv, keys);
    const std::string left_input_filename = parser.get<std::string>("left");
    const std::string right_input_filename = parser.get<std::string>("right");
    std::string out_filename = parser.get<std::string>("out");
    if (out_filename.empty())
        out_filename = "a_out";
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    if(!parser.check()) {
        parser.printErrors();
        return -1;
    }
    SPDLOG_INFO("processing {} as left and {} as right", left_input_filename, right_input_filename);



    cv::Mat dis;
    cv::Mat left = cv::imread(left_input_filename, cv::IMREAD_COLOR); //, 0); //read images into grayscale
    if (left.empty())
        SPDLOG_ERROR("left image {} not found", left_input_filename);
    SPDLOG_INFO("left image loaded.");
    cv::Mat right = cv::imread(right_input_filename, cv::IMREAD_COLOR); //, 0);
    if (right.empty())
        SPDLOG_ERROR("left right {} not found", right_input_filename);
    SPDLOG_INFO("right image loaded.");
    SPDLOG_INFO("images loaded.");
    if (left.rows > 450 || right.rows > 450) {
        SPDLOG_INFO("perform scaling.");
        left = scale_image(left);
        right = scale_image(right);
    }
    //dis = custom_stereomatching(left, right);
    dis = openCV_stereomatching(left,right);
    cv::imwrite(fmt::format("{}_depth.png",out_filename), dis);
    cv::namedWindow("disparity", cv::WINDOW_AUTOSIZE);
    cv::imshow("disparity", dis);
    SPDLOG_INFO("depth map written.");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = generate_pointcloud(dis);
    SPDLOG_INFO("point cloud generated.");
    pcl::io::savePLYFile(fmt::format("{}_depth.ply",out_filename),*cloud,true);
    SPDLOG_INFO("done.");
    return 0;
}