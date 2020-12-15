#include "ImageUtil.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <boost/lexical_cast.hpp>


bool UndistortKeyPoints(const cv::Mat &K, const cv::Mat &dist,
                        const std::vector<cv::Point2f> &src,
                        std::vector<cv::Point2f> &undistort) {
    if (dist.at<float>(0) == 0.0) {
        undistort = src;
        return false;
    }

    // Fill matrix with points
    int N = src.size();
    cv::Mat mat(N, 2, CV_32F);

    for (int i = 0; i < N; i++) {
        mat.at<float>(i, 0) = src[i].x;
        mat.at<float>(i, 1) = src[i].y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, dist, cv::Mat(), K);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    undistort.resize(N);

    for (int i = 0; i < N; i++) {
        cv::Point2f &kp = undistort[i];
        kp.x = mat.at<float>(i, 0);
        kp.y = mat.at<float>(i, 1);
    }
}


cv::Mat drawMatches(const cv::Mat &imk, const cv::Mat &imk_1,
                    const std::vector<cv::Point2f> &kpsk, const std::vector<cv::Point2f> &kpsk_1,
                    const std::vector<uchar> &status){
    cv::Mat imMatches;
    cv::hconcat(imk, imk_1, imMatches);
    cv::cvtColor(imMatches, imMatches, CV_GRAY2BGR);
    cv::RNG rng;
    for(int i = 0; i<kpsk.size(); i++){
        if(!status[i]) continue;

        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);
        // LOG(INFO) <<  kpsk[i] << " " << kpsk_1[i];
        cv::line(imMatches, kpsk[i], kpsk_1[i]+cv::Point2f(imk.cols, 0), cv::Scalar(b, g, r));
    }
    return imMatches;
}

cv::Mat skew(const cv::Mat &a)
{
    cv::Mat ax(cv::Mat::zeros(3,3,CV_32FC1));
    float x = a.at<float>(0);
    float y = a.at<float>(1);
    float z = a.at<float>(2);
    ax.at<float>(0, 1) = -z;
    ax.at<float>(0, 2) = y;
    ax.at<float>(1, 0) = z;
    ax.at<float>(1, 2) = -x;
    ax.at<float>(2, 0) = -y;
    ax.at<float>(2, 1) = x;
    return ax;
}

cv::Mat exp(const cv::Mat &theta){
    float phi = cv::norm(theta);
    if(phi<1e-3)
        return cv::Mat::eye(3,3,CV_32FC1);
    cv::Mat phiv = theta/phi;
    cv::Mat phix = skew(phiv);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1) + sin(phi)*phix+(1-cos(phi))*phix*phix;
    return R;
}

cv::Mat ConvertImageToUChar(const cv::Mat & image)
{
    cv::Mat im;
    image.convertTo(im, CV_8UC3, 255);
    return im;
}

cv::Mat DrawRect2(const cv::Mat &im1, const cv::Mat &im2, const std::vector<cv::Point2f> &srcPts, const std::vector<cv::Point2f> &dstPts){
    cv::Mat im1U, im2U;
    if(im1.type() != CV_8UC1)
         im1U = ConvertImageToUChar(im1);
    else
         im1U = im1.clone();

    if(im2.type() != CV_8UC1)
        im2U = ConvertImageToUChar(im2);
    else
        im2U = im2.clone();
    const int N = 4;
    auto modN = [&](int a){return a%N;};
    for (size_t i = 0; i<N; i++){
        cv::line(im2U, dstPts[modN(i)], dstPts[modN(i+1)], cv::Scalar(255,255,255), 3);
    }
    cv::Mat out;
    cv::hconcat(im1U, im2U, out);
    float iou = PolygonIntersection::IOU(srcPts, dstPts);
    char buf[256];
    sprintf(buf, "IOU: %.2f", iou);
    cv::putText(out, buf,
                cv::Point(15, out.rows - 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));
    return out;
}

bool PolygonIntersection::verbose = false;

// Returns x-value of point of intersectipn of two
// lines
float PolygonIntersection::x_intersect(float x1, float y1, float x2, float y2,
        float x3, float y3, float x4, float y4)
{
    float num = (x1*y2 - y1*x2) * (x3-x4) -
        (x1-x2) * (x3*y4 - y3*x4);
    float den = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4);
    CHECK_GT(fabs(den), 1e-6);
    return num/den;
}

// Returns y-value of point of intersectipn of
// two lines
float PolygonIntersection::y_intersect(float x1, float y1, float x2, float y2,
        float x3, float y3, float x4, float y4)
{
    float num = (x1*y2 - y1*x2) * (y3-y4) -
        (y1-y2) * (x3*y4 - y3*x4);
    float den = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4);
    CHECK_GT(fabs(den), 1e-6);
    return num/den;
}

// This functions clips all the edges w.r.t one clip
// edge of clipping area
void PolygonIntersection::clip(std::vector<cv::Point2f> &poly_points,
        float x1, float y1, float x2, float y2)
{
    std::vector<cv::Point2f> new_points;
    // (ix,iy),(kx,ky) are the co-ordinate values of
    // the points
    const float th = -1e-6;
    for (int i = 0, poly_size = poly_points.size(); i < poly_size; i++)
    {
        // i and k form a line in polygon
        int k = (i+1) % poly_size;
        float ix = poly_points[i].x, iy = poly_points[i].y;
        float kx = poly_points[k].x, ky = poly_points[k].y;

        // Calculating position of first point
        // w.r.t. clipper line
        float i_pos = (x2-x1) * (iy-y1) - (y2-y1) * (ix-x1);

        // Calculating position of second point
        // w.r.t. clipper line
        float k_pos = (x2-x1) * (ky-y1) - (y2-y1) * (kx-x1);

        // Case 1 : When both points are inside
        if (i_pos < th  && k_pos < th)
        {
            new_points.emplace_back(kx, ky);
        }
        // Case 2: When only first point is outside
        else if (i_pos > th  && k_pos <= th)
        {
            // Point of intersection with edge
            // and the second point is added
            float xnew = x_intersect(x1,
                    y1, x2, y2, ix, iy, kx, ky);
            float ynew = y_intersect(x1,
                    y1, x2, y2, ix, iy, kx, ky);
            new_points.emplace_back(xnew, ynew);
            new_points.emplace_back(kx, ky);
        }
        // Case 3: When only second point is outside
        else if (i_pos <= th  && k_pos > th)
        {
            //Only point of intersection with edge is added
            float xnew = x_intersect(x1,
                    y1, x2, y2, ix, iy, kx, ky);
            float ynew = y_intersect(x1,
                    y1, x2, y2, ix, iy, kx, ky);
            new_points.emplace_back(xnew, ynew);
        }
        // Case 4: When both points are outside
        else
        {
            //No points are added
        }
    }
    poly_points.swap(new_points);

    if (verbose){
        LOG(INFO) << "-----------";
        for (size_t i = 0, iend = poly_points.size(); i<iend; i++){
            LOG(INFO) << poly_points[i];
        }
    }
}

// Implements Sutherland–Hodgman algorithm
std::vector<cv::Point2f> PolygonIntersection::suthHodgClip(std::vector<cv::Point2f> poly_points, const std::vector<cv::Point2f> &clipper_points)
{
    //i and k are two consecutive indexes
    for (int i=0, clipper_size = poly_points.size(); i<clipper_size; i++)
    {
        int k = (i+1) % clipper_size;

        // We pass the current array of vertices, it's size
        // and the end points of the selected clipper line
        clip(poly_points, clipper_points[i].x,
                clipper_points[i].y, clipper_points[k].x,
                clipper_points[k].y);
    }

    // Printing vertices of clipped polygon
    if (verbose){
        for (int i=0, poly_size = poly_points.size(); i < poly_size; i++)
            LOG(INFO) << '(' << poly_points[i].x <<
                ", " << poly_points[i].y << ") ";

    }
    return poly_points;
}

float PolygonIntersection::IOU(const std::vector<cv::Point2f> &poly_points, const std::vector<cv::Point2f> &clipper_points){
    std::vector<cv::Point2f> pts = suthHodgClip(poly_points, clipper_points);
    float intersectArea = areaPolygon(pts);
    float polyArea = areaPolygon(poly_points);
    float clipperArea = areaPolygon(clipper_points);
    float den = clipperArea + polyArea - intersectArea;
    if (den < 1e-6){
        return 0.;
    }else{
        return intersectArea/den;
    }
}

float PolygonIntersection::areaPolygon(const std::vector<cv::Point2f> &poly_points){
    if (poly_points.size()<3) return 0.;
    float sArea = 0.;
    const cv::Point2f &origin = poly_points[0];
    for (size_t i = 1, iend = poly_points.size() - 1; i<iend; i++){
        const cv::Point2f &p1 = poly_points[i];
        const cv::Point2f &p2 = poly_points[i+1];
        sArea += areaTriangle(origin.x, origin.y, p1.x, p1.y, p2.x, p2.y);
    }
    return sArea;
}

float PolygonIntersection::areaTriangle(float dX0, float dY0, float dX1, float dY1, float dX2, float dY2)
{
    float dArea = ((dX1 - dX0)*(dY2 - dY0) - (dX2 - dX0)*(dY1 - dY0))/2.0;
    return (dArea > 0.0) ? dArea : -dArea;
}


Eigen::Matrix3d RotZT(double psi){
    Eigen::Matrix3d RDeltaPsi;
    RDeltaPsi.setIdentity();
    RDeltaPsi(0, 0) = cos(psi);
    RDeltaPsi(0, 1) = -sin(psi);
    RDeltaPsi(1, 1) = cos(psi);
    RDeltaPsi(1, 0) = sin(psi);
    return RDeltaPsi;
}

Eigen::Matrix3d RotZ(double psi){
    Eigen::Matrix3d RDeltaPsi;
    RDeltaPsi.setIdentity();
    RDeltaPsi(0, 0) = cos(psi);
    RDeltaPsi(0, 1) = sin(psi);
    RDeltaPsi(1, 1) = cos(psi);
    RDeltaPsi(1, 0) = -sin(psi);
    return RDeltaPsi;
}
