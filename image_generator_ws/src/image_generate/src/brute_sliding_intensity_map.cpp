#include "backend_common.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "cxcore.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

using std::signal;

typedef uint64_t TimeType;

std::atomic<bool> exit_flag;
float distance_threshold;
double point_screen_threshold_lower = 5;
double point_screen_threshold_upper = 50;


constexpr float loopClosureFreq = 1;
constexpr float mappingLeafSize = 0.15; // 0.2
constexpr float mappingGlobalMapSize = 0.1; // 0.1
constexpr int numberOfCores = 4;

const std::vector<float> zDividingValues {3, 6, 11};

pcl::VoxelGrid<PointType> downsizeFilterICP;

// cur
double timeLaserInfoCur; 
double timeLaserInfoLast;
float transformTobeMapped[6];

// ros
std::string odometryFrame = "odom";
nav_msgs::Path globalPath;
ros::Subscriber subOdom, subPointCloud;
ros::Publisher pubOdom, pubPath, pubContext;

std::mutex mutOdom;
std::deque<nav_msgs::Odometry::Ptr> odomQueue;
std::mutex mutPC;
std::deque<sensor_msgs::PointCloud2::Ptr> pointCloudQueue;

std::atomic<bool> newDataComing;
bool firstScanFlag = true;
gtsam::Pose3 lastOdomGtsam;
gtsam::Pose3 increPose;

gtsam::Pose3 curOdomGtsam;

#define COUT(s) std::cout << s << std::endl

void initVariables() {

}

void odomCallBack(const nav_msgs::Odometry::ConstPtr &msg) {
    nav_msgs::Odometry::Ptr pMsg(new nav_msgs::Odometry(*msg));
    std::unique_lock<std::mutex> lk(mutOdom);
    odomQueue.emplace_back(pMsg);
}

void pointCloudCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    sensor_msgs::PointCloud2::Ptr pMsg(new sensor_msgs::PointCloud2(*msg));
    std::unique_lock<std::mutex> lk(mutPC);
    pointCloudQueue.emplace_back(pMsg);
}

constexpr unsigned int SUBMAP_COUNT = 10;

struct VirtualCam {
    float k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
    uint height = 300, width = 300;
    float fx = 200, fy = 200, cx = 150.0, cy = 150.0;
    float maxIntensity = 0;

    gtsam::Pose3 pose;

    void setParam(uint _height, uint _width, float _fx, float _fy, float _cx, float _cy) {
        height = _height;
        width = _width;
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
    }

    void project(float x, float y, float z, int& u, int& v, bool& flag) {
        if (z == 0) {
            flag = false;
            return ;
        }
        x /= z; y /= z;
        float fu = fx*x + cx, fv = fy*y + cy;
        if (fu < 0 || fv < 0) {
            flag = false;
            return ;
        }
        int iu = int(fu + 0.5), iv = int(fv + 0.5);
        if (iu >= width || iv >= height) {
            flag = false;
            return ;
        }
        u = iu; v = iv;
        flag = true;
    }
    
};

float minIntensity = 0.001, maxIntensity = 80;

std::string prefixImgSavePath;
// std::string prefixMaskSavePath;
std::string prefixInpaintSavePath;
std::string camConfigPath;

VirtualCam virtualCam;
// std::vector<std::string> imgSavePaths;
// std::vector<cv::Mat> imgs;

// std::vector<std::vector<std::string>> imgSavePathsSUBMAPCOUNT;
// std::vector<std::vector<cv::Mat>> imgsSUBMAPCOUNT;

using ushort = unsigned short;

std::unordered_map<int, int> intensity_counter;

std::deque<PointCloudPtrType> slidingWindowPointClouds;
std::deque<gtsam::Pose3> slidingWindowPoses;
std::deque<TimeType> slidingWindowTimes;

// double notFillRate = 0, notFillRateCnt = 0;

void _project(PointCloudPtrType& ptr, VirtualCam& virtualCam, int idx, TimeType frameTime, bool useTimeAsImgName, bool saveImg) {
    static int nei[8][2] = {
        {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}
    };

    // double rateCnt = 0;
    
    std::string imgSavePath, maskSavePath, inpaintSavePath;
    if (useTimeAsImgName) {
        imgSavePath = prefixImgSavePath + std::to_string(frameTime) + ".png";
        // maskSavePath = prefixMaskSavePath + std::to_string(frameTime) + ".png";
        inpaintSavePath = prefixInpaintSavePath + std::to_string(frameTime) + ".png";
    } else {
        imgSavePath = prefixImgSavePath + std::to_string(idx) + ".png";
        // maskSavePath = prefixMaskSavePath + std::to_string(idx) + ".png";
        inpaintSavePath = prefixInpaintSavePath + std::to_string(idx) + ".png";
    }

    std::vector<std::vector<int>> imgCount(virtualCam.height, std::vector<int>(virtualCam.width, 0));
    std::vector<std::vector<double>> imgDis(virtualCam.height, std::vector<double>(virtualCam.width, 0));
    cv::Mat img = cv::Mat::zeros(cv::Size(virtualCam.width, virtualCam.height), CV_8UC1);

    std::sort(ptr->points.begin(), ptr->points.end(), [](const PointType& lhs, const PointType& rhs){
        return lhs.x*lhs.x + lhs.y*lhs.y + lhs.z*lhs.z > rhs.x*rhs.x + rhs.y*rhs.y + rhs.z*rhs.z;
    });

    for (uint i = 0; i < ptr->points.size(); ++i) {
        float x = -ptr->points[i].y,
              y = -ptr->points[i].z,
              z =  ptr->points[i].x;
        bool f = false;
        int u, v;
        virtualCam.project(x, y, z, u, v, f);

        if (f) {
            virtualCam.maxIntensity = 155;
            if (ptr->points[i].intensity > 155) {
                ptr->points[i].intensity = 155;
            };

            int &cnt = imgCount[v][u];
            uchar& oldVal = img.at<uchar>(v, u);
            
            ushort zVal = ushort(z * 2000);
            if (cnt != 0) {
                double oldDis = imgDis[v][u];
                double nowDis = std::sqrt(x*x + y*y + z*z);
                if (std::fabs(nowDis - oldDis) > 0.1) {
                    // getNearest
                    if (nowDis > oldDis) {

                    } else {
                        cnt = 1;
                        imgDis[v][u] = nowDis;

                        oldVal = uchar(ptr->points[i].intensity / virtualCam.maxIntensity * 255);
                    }
                } else {
                    // getAvg;
                    imgDis[v][u] = (imgDis[v][u] * cnt + nowDis) / (cnt + 1);
                    oldVal = uchar( (double(oldVal) * cnt + ptr->points[i].intensity / virtualCam.maxIntensity * 255) / (cnt+1));
                    ++cnt;
                }
            } else {
                double nowDis = std::sqrt(x*x + y*y + z*z);
                cnt = 1;
                oldVal = uchar(ptr->points[i].intensity / virtualCam.maxIntensity * 255);
                imgDis[v][u] = nowDis;
            }
        } 
    }
    
    cv::Mat imgMask = cv::Mat::zeros(cv::Size(virtualCam.width, virtualCam.height), CV_8UC1);
    for (uint u = 0; u < virtualCam.height; ++u) {
        for (uint v = 0; v < virtualCam.width; ++v) {
            if (!imgCount[u][v]) {
                bool hasProjectedNeighbor = false;
                int validNum = 0;
                int validVal = 0, validDepth = 0, validIntensity = 0;
                for (int i = 0; i < 8; ++i) {
                    int du = nei[i][0], dv = nei[i][1];
                    int nu = int(u) + du, nv = int(v) + dv;
                    if (nu >= 0 && nv >= 0 && nu < virtualCam.height && nv < virtualCam.width && imgCount[nu][nv]) {
                        hasProjectedNeighbor = true;
                        // validNum++;
                        validNum += imgCount[nu][nv];
                        validVal += int(img.at<uchar>(nu, nv)) * imgCount[nu][nv];
                    }
                }
                if (!hasProjectedNeighbor) {
                    // rateCnt += 1;
                    imgMask.at<uchar>(u, v) = 255;
                } else {
                    int naiveVal = float(validVal) / validNum;
                    img.at<uchar>(u, v) = uchar(naiveVal);
                }
            }
        }
    }

    // rateCnt /= virtualCam.width * virtualCam.height;
    // notFillRate = (notFillRateCnt*notFillRate + rateCnt) / (notFillRateCnt + 1);
    // notFillRateCnt++;
    
    if (true) {
        cv::Mat result_img;
        result_img = img;
        cv::inpaint(img, imgMask, result_img, 3, cv::INPAINT_TELEA);
        if (saveImg) {
            cv::imwrite(imgSavePath, img);
            cv::imwrite(inpaintSavePath, result_img);
        }
    }

    // publish
}

void projectIntensityImage(gtsam::Pose3 projectPose, TimeType projectTime, bool saveImg) {
    assert(slidingWindowPointClouds.size() == SUBMAP_COUNT);
    
    static int cnt = 0;
    virtualCam.pose = projectPose;
    PointPoseType poseBetween = gtsamPose2pcl(virtualCam.pose.between(slidingWindowPoses.back()));
    PointCloudPtrType ptr(new PointCloudType());

    *ptr = *transformPointCloud(slidingWindowPointClouds.back(), &poseBetween);
    bool useTimeAsImgName = true;

    sensor_msgs::PointCloud2::Ptr pContext(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ptr, *pContext);
    pContext->header.frame_id = "camera_init";
    pubContext.publish(pContext);

    _project(ptr, virtualCam, cnt++, projectTime, useTimeAsImgName, saveImg);

}

void eliminatePoints(PointCloudPtrType& ptrEliminate, PointCloudPtrType& ptrNow) {
    for (auto p: ptrNow->points) {
        if (p.x*p.x + p.y*p.y + p.z*p.z >= point_screen_threshold_lower*point_screen_threshold_lower
            && p.x*p.x + p.y*p.y + p.z*p.z <= point_screen_threshold_upper*point_screen_threshold_upper) {
            ptrEliminate->push_back(p);
        }
    }
}

void work() {
    initVariables();
    while (ros::ok()) {
        if (exit_flag) return;
        sensor_msgs::PointCloud2::Ptr pCurPointCloudROS(new sensor_msgs::PointCloud2());
        nav_msgs::Odometry::Ptr pCurOdom(new nav_msgs::Odometry());
        if (!odomQueue.empty() && !pointCloudQueue.empty()) {

            {
                std::unique_lock<std::mutex> lkOdom(mutOdom), lkPC(mutPC);
                while (!odomQueue.empty() && odomQueue.front()->header.stamp < pointCloudQueue.front()->header.stamp) odomQueue.pop_front();
                while (!pointCloudQueue.empty() && pointCloudQueue.front()->header.stamp < pointCloudQueue.front()->header.stamp) pointCloudQueue.pop_front();
                if (odomQueue.empty() || pointCloudQueue.empty() || odomQueue.front()->header.stamp != pointCloudQueue.front()->header.stamp) {
                    newDataComing = false;
                    continue;
                }
                newDataComing = true;
                pCurPointCloudROS = pointCloudQueue.front(); pointCloudQueue.pop_front();
                pCurOdom = odomQueue.front(); odomQueue.pop_front();
            }
            timeLaserInfoCur = pCurOdom->header.stamp.toSec();   
            PointCloudPtrType ptrNow(new PointCloudType());
            pcl::fromROSMsg(*pCurPointCloudROS, *ptrNow);
            curOdomGtsam = gtsam::Pose3(
                gtsam::Rot3::Quaternion(pCurOdom->pose.pose.orientation.w,
                                        pCurOdom->pose.pose.orientation.x,
                                        pCurOdom->pose.pose.orientation.y,
                                        pCurOdom->pose.pose.orientation.z),
                gtsam::Point3( pCurOdom->pose.pose.position.x,
                                pCurOdom->pose.pose.position.y,
                                pCurOdom->pose.pose.position.z)
            );
            
            PointCloudPtrType ptrEliminate(new PointCloudType());
            eliminatePoints(ptrEliminate, ptrNow);

            TimeType curTime = pCurOdom->header.stamp.toNSec();
            if (slidingWindowPointClouds.size() < SUBMAP_COUNT) {
                slidingWindowPointClouds.emplace_front(ptrEliminate);
                slidingWindowPoses.emplace_front(curOdomGtsam);
                slidingWindowTimes.emplace_front(curTime);

                for (int i = 1; i < slidingWindowPointClouds.size(); ++i) {
                    gtsam::Pose3 poseBetween = slidingWindowPoses[i].between(curOdomGtsam);
                    auto poseBetweenPCL = gtsamPose2pcl(poseBetween);
                    *slidingWindowPointClouds[i] += *transformPointCloud(ptrEliminate, &poseBetweenPCL);
                }
            } else {
                static int frameCnt = 0;

                static bool firstFlag = true;
                static gtsam::Pose3 lastOdomGtsam;
                if (firstFlag) {
                    firstFlag = false;
                    lastOdomGtsam = curOdomGtsam;
                } else {
                    auto translation = lastOdomGtsam.between(curOdomGtsam).translation();
                    float distance_square = translation.x()*translation.x() + translation.y()*translation.y() + translation.z()*translation.z();
                    if (distance_square < distance_threshold*distance_threshold) continue;
                    lastOdomGtsam = curOdomGtsam; 
                }
                
                static std::chrono::nanoseconds timeCost = std::chrono::nanoseconds::zero();
                tictoc timer;

                slidingWindowPointClouds.emplace_front(ptrEliminate);
                slidingWindowPoses.emplace_front(curOdomGtsam);
                slidingWindowTimes.emplace_front(curTime);
                slidingWindowPointClouds.pop_back();
                slidingWindowPoses.pop_back();
                slidingWindowTimes.pop_back();

                for (int i = 1; i < slidingWindowPointClouds.size(); ++i) {
                    gtsam::Pose3 poseBetween = slidingWindowPoses[i].between(curOdomGtsam);
                    auto poseBetweenPCL = gtsamPose2pcl(poseBetween);
                    *slidingWindowPointClouds[i] += *transformPointCloud(ptrEliminate, &poseBetweenPCL);
                }

                const static int ProjectIdx = SUBMAP_COUNT / 2;

                bool saveImg = true;
                frameCnt++;

                if (frameCnt % 10 == 0) {
                    projectIntensityImage(slidingWindowPoses[ProjectIdx], slidingWindowTimes[ProjectIdx], saveImg);
                }

                timeCost += timer.toc();
                
                // if (frameCnt % 100 == 0) {
                //     std::cout << "100 frames costs: " <<  timeCost.count() << "ns" << " " << "rate: " << notFillRate << std::endl;
                //     timeCost = std::chrono::nanoseconds::zero();
                // }

            }
        } else {
            std::this_thread::sleep_for(std::chrono::steady_clock::duration(1000000));
        }
    }
}

void SigHandle(int sig) {
    exit_flag = true;
    ROS_WARN("catch sig%d", sig);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "intensity_submap");
    ros::NodeHandle nh;

    pubContext = nh.advertise<sensor_msgs::PointCloud2>("/context", 2);
    image_transport::ImageTransport it(nh);
    subOdom = nh.subscribe("/Odometry", 100000, odomCallBack);
    subPointCloud = nh.subscribe("/cloud_registered_body", 100000, pointCloudCallBack);

    cv::FileStorage fs("/home/mason/fox/projects/slam/image_generator_ws/src/image_generate/cfg/virtualCamParam.yaml", cv::FileStorage::READ);
    int height, width;
    float fx, fy, cx, cy;
    fs["height"] >> height;
    fs["width"] >> width;
    fs["fx"] >> fx;
    fs["fy"] >> fy;
    fs["cx"] >> cx;
    fs["cy"] >> cy;
    fs["pointThreshLower"] >> point_screen_threshold_lower;
    fs["pointThreshUpper"] >> point_screen_threshold_upper;
    virtualCam.setParam(height, width, fx, fy, cx, cy);
    fs["distance_threshold"] >> distance_threshold;
    fs["img_save_path"] >> prefixImgSavePath;
    fs["inpaint_save_path"] >> prefixInpaintSavePath;

    exit_flag = false;
    signal(SIGINT, SigHandle);

    std::thread workingThread {work};

    ros::Rate rate(5000);
    while (!exit_flag) {
        ros::spinOnce();
        rate.sleep();
    }

    workingThread.join();

    return 0;
}
