//
// Created by zsk on 18-11-25.
//

#ifndef OF_VELOCITY_STATE_H
#define OF_VELOCITY_STATE_H

#include "Sensor.h"
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "NormalVectorElement.h"
class PGOYawPosMeasurement{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int i1;
    int i2;
    double yaw;
    Eigen::Vector3d r12;
    PGOYawPosMeasurement();
    PGOYawPosMeasurement(int i1, int i2, int yaw, const Eigen::Vector3d &r12);

    PGOYawPosMeasurement(int i1, int i2);

};

class Frame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef NormalVectorElement<double> NormalVector;
    typedef typename NormalVector::QPD Quaternion;
    Eigen::Vector3d mNormal, mPos;
    Quaternion mRot;
    double mdInverseDistance;
    cv::Mat mIx, mIy, mImg;
    int mnFrameId, mnTrackedCandidate, mnTrackedSuccess;
    bool mbBad;
    std::vector<cv::Point2f> mTracks, mFeatures;

    Frame();
    Frame(const Frame& f);
    Frame(const cv::Mat &IKF, const cv::Mat &IxKF, const cv::Mat &IyKF, const Eigen::Vector3d &pos, const Eigen::Vector3d &n, const Quaternion &q, double alpha, int fid);
};

template <typename Scalar, int D>
class State{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef NormalVectorElement<Scalar> NormalVector;
    typedef typename NormalVector::QPD Quaternion;
    typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;
    typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
    typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

    typedef Eigen::Matrix<Scalar, D, D> MatrixState;
    typedef MatrixState MatrixIMUNoise;
    typedef Eigen::Matrix<Scalar, D, 1> VectorState;
    static const int stateDim = D;

    State(Scalar alpha, const Vector3 &p, const Vector3 &v, const typename NormalVector::QPD &rot, const Vector2 &n,
            const MatrixIMUNoise &mStateNoise, const MatrixState &P0);

    void PropagationIMUVIO(const std::vector<IMU> &imu);
    void PropagationIMUVIO(const IMU &imu);

    void MeasurementUpdateKFLastMarginalization(const std::vector<IMU> &imus, const cv::Mat &imLast, const cv::Mat &imCur, Scalar dt);

    void MeasurementUpdateKFLastLK(const std::vector<IMU> &imus, const cv::Mat &imLast, const cv::Mat &imCur, Scalar dt);

    void HomograpyMapImagePoints(const Matrix3 &H, std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst, int rows, int cols);

    cv::Mat DrawRect(const cv::Mat &im1, const cv::Mat &im2, const Matrix3 &H);

    Scalar GradientMagRMS(const cv::Mat &Ix, const cv::Mat &Iy);

    int SearchOverlapKF();

    float OptimizeCF(Frame* pKFLoop, Frame* pKFCur, float scale = 1.);

    float AverageIntensityChi2(int kfid);

    void YawPosMeasurement(const typename NormalVector::QPD &q1, const typename NormalVector::QPD &q2, const Vector3 &p1, const Vector3 &p2, Scalar &yaw, Vector3 &r12);

    Vector3 mPos, mBa, mBw, mVel;
    NormalVector mUnitDirection;
    typename NormalVector::QPD mRot;
    Scalar mfAlpha, mfCurentTime, mA, mB, mLastIOU;
    MatrixState mP, mStateNoise;
    Matrix3 mK, mKinv, mRci;
    cv::Mat mCVK, mCVKinv, mCVDist, remap1l_, remap2l_;
    cv::Mat Ixk_1, Iyk_1, imk_1;
    cv::Mat imk, Ixk, Iyk;
    int mnFrameId, mnLastKFIdx;
    std::vector<Frame*> mvpKfs;
    std::vector<PGOYawPosMeasurement, Eigen::aligned_allocator<PGOYawPosMeasurement> > mvMeasurement;
    Frame mKF;
    std::vector<cv::Point2f> mLastKFTracks;
	bool mbInit, mbHuber, mbLogDebug, mbInitKFRot;
};

#include "State.hpp"
// typedef State<double, 16> StateD16;
typedef State<double, 18> StateD18;
// typedef State<double, 20> StateD20;
#endif //OF_VELOCITY_STATE_H
