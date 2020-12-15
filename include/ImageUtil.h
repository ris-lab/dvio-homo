#ifndef IMAGE_UTIL_H_
#define IMAGE_UTIL_H_

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <vector>

// take from tracking repo: https://github.com/baidu-robotic-vision/tracking/blob/master/XP/src/util/image_utils.cc

// Compute the histogram of a sampled area of the input image and return the number of
// sampled pixels
int sampleBrightnessHistogram(const cv::Mat& raw_img,
                              std::vector<int>* histogram,
                              int* avg_pixel_val_ptr);

// [NOTE] Instead of matching the cdf(s), we brute-force scale the histograms and match them
// directly.  This matchingHistogram is intended to match two histograms of images taken with
// different gain/exposure settings.
float matchingHistogram(const std::vector<int>& hist_src,
                        const std::vector<int>& hist_tgt,
                        const float init_scale);

int drawHistogram(cv::Mat* img_hist,
                  const std::vector<int>& histogram,
                  bool auto_scale = false);

bool UndistortKeyPoints(const cv::Mat &K, const cv::Mat &dist,
                        const std::vector<cv::Point2f> &src,
                        std::vector<cv::Point2f> &undistort);


cv::Mat drawMatches(const cv::Mat &imk, const cv::Mat &imk_1,
                    const std::vector<cv::Point2f> &kpsk, const std::vector<cv::Point2f> &kpsk_1,
                    const std::vector<uchar> &status);

cv::Mat skew(const cv::Mat &a);

void MatchBrightness(const cv::Mat& tgt, cv::Mat* src);

template <typename Derived>
inline Eigen::Matrix<Derived, 3, 3> skew(const Eigen::Matrix<Derived, 3, 1> &a)
{
    Eigen::Matrix<Derived, 3, 3> ax;
    ax.setZero();
    Derived x = a(0);
    Derived y = a(1);
    Derived z = a(2);
    ax(0, 1) = -z;
    ax(0, 2) = y;
    ax(1, 0) = z;
    ax(1, 2) = -x;
    ax(2, 0) = -y;
    ax(2, 1) = x;
    return ax;
}

template <typename Derived>
void RotationDecomposition(const Eigen::Matrix<Derived, 3, 3> &Rcw, Eigen::Matrix<Derived, 3, 3> &Rn, Eigen::Matrix<Derived, 3, 3> &Rpsi){
    Eigen::Matrix<Derived, 3, 1> n = Rcw.col(2);
    Rn(0, 0) = 1 - n(0)*n(0)/(1+n(2));
    Rn(0, 1) = -n(0)*n(1)/(1+n(2));
    Rn(0, 2) = n(0);
    Rn(1, 0) = -n(0)*n(1)/(1+n(2));
    Rn(1, 1) = 1 - n(1)*n(1)/(1+n(2));
    Rn(1, 2) = n(1);
    Rn(2, 0) = -n(0);
    Rn(2, 1) = -n(1);
    Rn(2, 2) = n(2);
    Rpsi = Rn.transpose()*Rcw;
}

Eigen::Matrix3d RotZT(double psi);

Eigen::Matrix3d RotZ(double psi);

cv::Mat ConvertImageToUChar(const cv::Mat & image);

cv::Mat DrawRect2(const cv::Mat &im1, const cv::Mat &im2, const std::vector<cv::Point2f> &srcPts, const std::vector<cv::Point2f> &dstPts);


inline bool InsideImage(float px, float py, int rows, int cols){
    return (px >= 0. && px <= static_cast<float>(cols) - 1.0001
            && py >= 0. && py <= static_cast<float>(rows) - 1.0001);
}

class PolygonIntersection{
public:
    static bool verbose;

    // Returns x-value of point of intersectipn of two
    // lines
    static float x_intersect(float x1, float y1, float x2, float y2,
            float x3, float y3, float x4, float y4);

    // Returns y-value of pofloat of floatersectipn of
    // two lines
    static float y_intersect(float x1, float y1, float x2, float y2,
            float x3, float y3, float x4, float y4);

    // This functions clips all the edges w.r.t one clip
    // edge of clipping area
    static void clip(std::vector<cv::Point2f> &poly_points,
        float x1, float y1, float x2, float y2);

    // Implements Sutherland–Hodgman algorithm
    static std::vector<cv::Point2f> suthHodgClip(std::vector<cv::Point2f> poly_points, const std::vector<cv::Point2f> &clipper_points);

    static float IOU(const std::vector<cv::Point2f> &poly_points, const std::vector<cv::Point2f> &clipper_points);

    static float areaPolygon(const std::vector<cv::Point2f> &poly_points);

    static float areaTriangle(float dX0, float dY0, float dX1, float dY1, float dX2, float dY2);

};
#endif
