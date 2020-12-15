#include "State.h"

Frame::Frame():
    mNormal(0, 0, 1.), 
    mPos(0, 0, 0),
    mdInverseDistance(1.),
    mnFrameId(0),
    mnTrackedCandidate(0), 
    mnTrackedSuccess(0),
    mbBad(false)
{

}

Frame::Frame(const cv::Mat &IKF, const cv::Mat &IxKF, const cv::Mat &IyKF, const Eigen::Vector3d &pos, const Eigen::Vector3d &n, const Quaternion &q, double alpha, int fid):
mImg(IKF), mIx(IxKF), mIy(IyKF), 
    mNormal(n), mPos(pos), mRot(q),
mdInverseDistance(alpha), mnFrameId(fid),
mnTrackedCandidate(0), mnTrackedSuccess(0), mbBad(false){

}

Frame::Frame(const Frame &f):
mImg(f.mImg.clone()), mIx(f.mIx.clone()), mIy(f.mIy.clone()), 
    mNormal(f.mNormal), mPos(f.mPos), mRot(f.mRot),
mdInverseDistance(f.mdInverseDistance), mnFrameId(f.mnFrameId),
mnTrackedCandidate(0), mnTrackedSuccess(0), mbBad(false),
mTracks(f.mTracks), mFeatures(f.mFeatures){


}

PGOYawPosMeasurement::PGOYawPosMeasurement():
i1(0), i2(0), yaw(0.), r12(0., 0., 0.){

}

PGOYawPosMeasurement::PGOYawPosMeasurement(int i1, int i2, int yaw, const Eigen::Vector3d &r12):
i1(i1), i2(i2), yaw(yaw), r12(r12){

}

PGOYawPosMeasurement::PGOYawPosMeasurement(int i1, int i2):
i1(i1), i2(i2), yaw(0.), r12(0., 0., 0.){

}




