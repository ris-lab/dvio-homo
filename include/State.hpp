//
// Created by zsk on 18-11-25.
//

#include <glog/logging.h>
#include "setting_config.h"
#include "Utilities.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include "vio_macro.h"
#include "ImageUtil.h"
//#define DRAW_OPTICAL_FLOW

#define VALUE_FROM_ADDRESS(address) (*(address))

template <typename Scalar, int _D>
State<Scalar, _D>::State(Scalar alpha, const Vector3 &p, const Vector3 &v, const typename NormalVector::QPD &rot, const Vector2 &n,
                         const MatrixIMUNoise &mStateNoise, const MatrixState &P0):
mfAlpha(alpha), mPos(p), mVel(v), mRot(rot), mUnitDirection(
        Vector3(n(0), n(1), sqrt(1 - n(0)*n(0) - n(1)*n(1)))),
mBa(0, 0, 0.), mBw(0, 0, 0),
mP(P0),
mStateNoise(mStateNoise), mfCurentTime(0.f),
mbInit(false),
mbInitKFRot(false),
mbHuber(g_use_huber),
mbLogDebug(g_show_log > 0),
mnFrameId(0), mnLastKFIdx(-1),
mA(1.),
mB(0.),
mLastIOU(1.)
{
    mK.setIdentity();
//    const float s = 1.f/(1<<g_level);
    const float s = g_scale;

    mCVK = cv::Mat::eye(3,3,CV_32FC1);
    mCVDist = (cv::Mat_<float>(4,1) << g_k1, g_k2, g_p1, g_p2);
    mCVK.at<float>(0, 0) = g_fx*s;
    mCVK.at<float>(1, 1) = g_fy*s;
    mCVK.at<float>(0, 2) = g_cx*s;
    mCVK.at<float>(1, 2) = g_cy*s;
    mCVKinv = mCVK.inv();

    cv::Mat mNewK;
    if(g_k1 != 0)
    {
        //mNewK = cv::getOptimalNewCameraMatrix(mCVK, mCVDist, cv::Size(g_frame_cols*s, g_frame_rows*s), 0);
        mNewK = cv::Mat::eye(3, 3, CV_32FC1);
        mNewK.at<float>(0, 0) = g_new_fx*s;
        mNewK.at<float>(1, 1) = g_new_fy*s;
        mNewK.at<float>(0, 2) = g_new_cx*s;
        mNewK.at<float>(1, 2) = g_new_cy*s;
        LOG(INFO) << "new K: " << mNewK ;
        if (!g_fisheye){
            cv::initUndistortRectifyMap(mCVK, mCVDist, cv::Mat(), mNewK,
                    cv::Size(g_frame_cols*s, g_frame_rows*s), CV_16SC2, remap1l_, remap2l_);
        }else{
            cv::fisheye::initUndistortRectifyMap(mCVK, mCVDist, cv::Mat(), mNewK,
                    cv::Size(g_frame_cols*s, g_frame_rows*s), CV_16SC2, remap1l_, remap2l_);
        }
    }

    if(!mNewK.empty())
    {
        mK(0, 0) = mNewK.at<float>(0, 0);
        mK(1, 1) = mNewK.at<float>(1, 1);
        mK(0, 2) = mNewK.at<float>(0, 2);
        mK(1, 2) = mNewK.at<float>(1, 2);
        mKinv = mK.inverse();
    } else
    {
        mK(0, 0) = g_fx*s;
        mK(1, 1) = g_fy*s;
        mK(0, 2) = g_cx*s;
        mK(1, 2) = g_cy*s;
        mKinv = mK.inverse();
    }
    Eigen::Matrix3f Rci;
    cv::cv2eigen(g_Tci.rowRange(0, 3).colRange(0, 3), Rci);
    mRci = Rci.cast<Scalar>();
    LOG(INFO) << "state noise: " << mStateNoise;
}

template <typename Scalar, int D>
void State<Scalar, D>::PropagationIMUVIO(const std::vector<IMU> &imus){
    for(const auto &imu:imus)
        PropagationIMUVIO(imu);
}

template <typename Scalar, int _D>
void State<Scalar, _D>::PropagationIMUVIO(const IMU &imu){
    if(!mbInit)
    {
        mfCurentTime = imu.timestamp;
        mbInit = true;
    }
    assert(imu.timestamp >= mfCurentTime);
    Scalar dt = imu.timestamp - mfCurentTime;
    Vector3 acc, gyro;

    for(int i = 0; i<3; i++)
    {
        acc(i) = imu.acc[i] - mBa(i);
        gyro(i) = imu.gyro[i]- mBw(i);
    }
    if (!mbInitKFRot){
        typename NormalVector::V3D a(0, 0, 1);
        Vector3 b = -acc/acc.norm();
        typename NormalVector::V3D rot = NormalVector::getRotationFromTwoNormals(b, a);
        mRot = mRot.exponentialMap(rot);
        LOG(INFO) << "kf rotation: " << rot
        << " rot mat: " << typename NormalVector::MPD(mRot).matrix() << " b: " << b.transpose();
        mbInitKFRot = true;
        mKF.mRot = mRot;
    }
    bool use_first_order = true;
    /*update state estimate*/
    const Vector3 g0(0, 0, -GRAVITY);
    Matrix3 Rwc = typename NormalVector::MPD(mRot).matrix();
    Matrix3 Rcw = Rwc.transpose();
    Vector3 g = Rcw*g0;
    Vector3 n = mUnitDirection.getVec();
    Vector3 vW = Rwc*mVel;
    typename NormalVector::V3D dm = dt*gyro;
    typename NormalVector::QPD qm = qm.exponentialMap(dm);

    if(mbLogDebug)
        LOG(INFO) << "dt : " << dt << " ori acc: " << acc.transpose() << " acc: " <<
                  (acc - g).transpose() << " gyro: " << gyro.transpose() << " n: " << n.transpose() << " g: " << g.transpose() << " Rcw: " << Rcw;

    Matrix3 dR = typename NormalVector::MPD(qm).matrix();
    Matrix3 dRT = dR.transpose();

    Matrix3 skewGyro = skew(gyro);
    Vector3 pos = mPos + vW*dt;
    Vector3 vel;
    if (use_first_order){
        vel = mVel + (-skewGyro*mVel + acc - g)*dt;
    }else{
        vel = dRT*mVel + (acc - g)*dt;
    }
    typename NormalVector::QPD qwc = mRot*qm;
    Scalar nTv = n.dot(mVel);
    Scalar alpha = mfAlpha * (1 + mfAlpha*nTv*dt);

    // update normal
    Matrix3 I3;
    I3.setIdentity();
    typename NormalVector::V3D dm2 = -dt*(I3-n*n.transpose())*gyro;
    typename NormalVector::QPD qm2 = qm2.exponentialMap(dm2);
    NormalVector nOut = mUnitDirection.rotated(qm2);
    //LOG(INFO) << "nout: " << nOut.getVec().transpose() << " nin: " << mUnitDirection.getVec() << " dm2: " << dm2.transpose() << " gyro: " << gyro.transpose();

    /*update cov*/
    // error propagatoin
    MatrixState F;
    F.setIdentity();
    Eigen::Matrix<Scalar, 3, 2> M = mUnitDirection.getM();
    Eigen::Matrix<Scalar, 2, 3> G23 = nOut.getM().transpose()*NormalVector::gSM(qm2.rotate(n))*NormalVector::Lmat(dm2)*(I3-n*n.transpose());

    //jac of position
    F.template block<3, 3>(0, 3) = Rwc*dt; // vel
    F.template block<3, 3>(0, 6) = -dt*skew(vW); // orientation

    // jac of velocity
    if(use_first_order){
        F.template block<3, 3>(3, 3) += dRT*dt; // vel
        F.template block<3, 3>(3, 6) = -dt*dRT*Rcw*skew(g0); // orientation
        F.template block<3, 3>(3, 9) = -dt*(I3+dRT)/2.;   // ba
    }else{
        F.template block<3, 3>(3, 3) += -skewGyro*dt; // vel
        F.template block<3, 3>(3, 6) = -dt*Rcw*skew(g0); // orientation
        F.template block<3, 3>(3, 9) = -dt*I3;   // ba
    }
    F.template block<3, 3>(3, 12) = -dt*skew(mVel);   // bw
    //jac of attitude
    //F.template block<3, 3>(6, 6) = I3; // orientation
    //F.template block<3, 3>(6, 6) = dRT; // orientation
    Matrix3 RcwLdm = Rwc*NormalVector::Lmat(dm);
    F.template block<3, 3>(6, 12) = -dt*RcwLdm;   // bw
    // jac of ba identity F.template block<3, 3>(9, 9)
    // jac of bw identity F.template block<3, 3>(12, 12)

    // jac of altitude
    F.template block<1, 3>(15, 3) = dt*mfAlpha*mfAlpha*n.transpose();   // vel
    F.template block<1, 2>(15, 16) = dt*mfAlpha*mfAlpha*mVel.transpose()*M;   // the normal 
    F(15, 15) += 2*mfAlpha*nTv*dt;  // alpha

    // jac of normal vector
    Vector3 dw = -gyro*dt;
    F.template block<2, 2>(16, 16) = nOut.getM().transpose()*(
            NormalVector::gSM(qm2.rotate(n))*NormalVector::Lmat(dm2)*(
                    -(I3*n.dot(dw)+n*dw.transpose()))
            +typename NormalVector::MPD(qm2).matrix()
    )*M; // normal
    F.template block<2, 3>(16, 12) = G23*dt; // bw

    if(mbLogDebug)
        LOG(INFO) << "F = " << F;

    Scalar sqrtime = sqrt(dt);
    Eigen::Matrix<Scalar, _D, _D> G;
    G.setIdentity();
    G.template block<3, 3>(0, 0) *= sqrtime;    //pos
    G.template block<3, 3>(3, 3) *= sqrtime;    //vel
    G.template block<3, 3>(3, 6) = -skew(mVel)*sqrtime;    //vel w.r.t att
    G.template block<3, 3>(6, 6) = RcwLdm*sqrtime; //att
    G.template block<3, 3>(9, 9) *= sqrtime; //ba
    G.template block<3, 3>(12, 12) *= sqrtime; //bw
    G(15, 15) *= sqrtime; //alpha
    G.template block<2, 2>(16, 16) = G23*mUnitDirection.getN()*sqrtime; //normal
    //G.template block<2, 2>(16, 16) = G23*mUnitDirection.getM()*sqrtime; //normal
    if(mbLogDebug)
        LOG(INFO) << "G = " << G;

    mP = F*mP*F.transpose() + G*mStateNoise*G.transpose();
    mP = (mP+mP.transpose())/2.;
    if(mbLogDebug){
        LOG(INFO) << "mP = " << mP;
        //LOG(INFO) << ""
    }

    mPos = pos;
    mVel = vel;
    mRot = qwc;
    mfAlpha = alpha;
    mUnitDirection.q_ = nOut.q_;

    mfCurentTime = imu.timestamp;
    if(mbLogDebug)
        LOG(INFO) << "pos = " << mPos.transpose() << " vel = " << mVel.transpose() << " alpha = " << mfAlpha;
}

template <typename Scalar, int _D>
void State<Scalar, _D>::MeasurementUpdateKFLastLK(const std::vector<IMU> &imus, const cv::Mat &imLast, const cv::Mat &imCur, Scalar dt){
    Vector3 gyro(0, 0, 0);
    for(const IMU& imu:imus)
    {
        gyro(0) += imu.gyro[0];
        gyro(1) += imu.gyro[1];
        gyro(2) += imu.gyro[2];
        if(mbLogDebug)
            LOG(INFO) << "gyro: " << imu.gyro[0] << " " << imu.gyro[1] << " " << imu.gyro[2];
    }
    gyro = 1.f/imus.size()*gyro;
    const float s = g_scale;
    if(mbLogDebug)
    {
        LOG(INFO) << "gyro = " << gyro.transpose();
        LOG(INFO) << "K = " << mK << " Kinv = " << mKinv << " dt: " << dt;
    }
    if(imk_1.empty())
    {
        cv::resize(imLast, imk_1, cv::Size(), s, s);
        if (g_use_median_blur)
            cv::medianBlur(imk_1, imk_1, 3);
        else
            imk_1 = HomoAlign::SmoothImage(g_gaussian_sigma, imk_1);
        if(g_k1 != 0) {
            cv::remap(imk_1, imk_1, remap1l_, remap2l_, cv::INTER_LINEAR);
        }
        imk_1.copyTo(mKF.mImg);
        mKF.mnFrameId = 0;
        mnFrameId = 1;
        //mnKFId = 0;
    }

    //cv::Mat imk;
    cv::resize(imCur, imk, cv::Size(), s, s);
    if (g_use_median_blur)
        cv::medianBlur(imk, imk, 3);
    else
        imk = HomoAlign::SmoothImage(g_gaussian_sigma, imk);

    if(g_k1 != 0)
    {
        cv::remap(imk, imk, remap1l_, remap2l_, cv::INTER_LINEAR);
        //cv::imwrite("undis.png", imk);
    }
	//static int nimg = 0 ;
	//char buf[256];
	//sprintf(buf, "/tmp/%04d.png", nimg++);
	//cv::imwrite(buf, imk);


    if(mbLogDebug){
        cv::imshow("Ik", imCur);
        cv::waitKey(1);
    }

    std::vector<cv::Point2f> kpsk, kpsk_1, kpskkf;
    cv::goodFeaturesToTrack(imk, kpsk, 50, 0.1, 10);
    std::vector<uchar> status, statuskf;
    std::vector<float> err, errkf;
    int levels = 3, win = 20;
    cv::calcOpticalFlowPyrLK(imk, imk_1, kpsk, kpsk_1, status, err, cv::Size(win, win), levels);
    if (mnFrameId > g_init_nkf && (g_use_kf == 1 || g_use_kf == 2))
        cv::calcOpticalFlowPyrLK(imk_1, imk, mLastKFTracks, kpskkf, statuskf, errkf, cv::Size(win, win), levels);

    Vector3 ez(0, 0, 1.);
    Matrix3 I3;
    I3.setIdentity();

    int iter = 0;
    int maxIter = g_max_iteration;

    Scalar alpha0 = mfAlpha;
    Vector3 ba0 = mBa;
    Vector3 bw0 = mBw;
    Vector3 pos0 = mPos;
    Vector3 vel0 = mVel;
    typename NormalVector::QPD rot0 = mRot;
    if (mbLogDebug){
        LOG(INFO) << "-------------------------------------";
    }
    NormalVector normal0;
    normal0.q_ = mUnitDirection.q_;
    const Scalar &fx = mK(0, 0);
    const Scalar &fy = mK(1, 1);
    const Scalar &cx = mK(0, 2);
    const Scalar &cy = mK(1, 2);
    Matrix3 RKFw = typename NormalVector::MPD(mKF.mRot).matrix().transpose();
    Vector3 PosKF = -RKFw*mKF.mPos;

    while (true) {
        START_CV_TIME(tIteration);
        Vector3 n, gyro_b;
        n = mUnitDirection.getVec();
        const Scalar &nx = n(0);
        const Scalar &ny = n(1);
        const Scalar &nz = n(2);
        const Scalar &vx = mVel(0);
        const Scalar &vy = mVel(1);
        const Scalar &vz = mVel(2);

        gyro_b = gyro - mBw;
        Eigen::Matrix<Scalar, 3, 2> N = mUnitDirection.getM();
        Matrix3 Rwc = typename NormalVector::MPD(mRot).matrix();
        Matrix3 RKFc = RKFw*Rwc;
        Vector3 posKFc = RKFw*mPos + PosKF;
        Matrix3 Hkf = mK*(RKFc+mfAlpha*posKFc*n.transpose())*mKinv;
        if (mbLogDebug)
            LOG(INFO) << "RKFw: " << RKFw << " RKFc: " << RKFc << " pos: " << mPos.transpose() << " n: " << n.transpose() << " mRot: " << mRot << " alpha: " << mfAlpha << " poseKFc: " << posKFc;

        MatrixState imHessian;
        imHessian.setZero();
        VectorState JTe;
        JTe.setZero();

        Scalar chi2 = 0.f;
        Scalar chi2KF = 0.f;
        int border = 3;
        int step = 1;

        int npixel = 0;
        int npixelKF = 0;
        Matrix3 H = mK*(skew(gyro_b) + mfAlpha*mVel*n.transpose())*mKinv;

        START_CV_TIME(tHJTe);
        float gradientKF = 0.;
        for (int i = 0, iend = kpsk.size(); i<iend; i++){
            if(!status[i])
                continue;
            Scalar x = kpsk[i].x;
            Scalar y = kpsk[i].y;

            const Vector3 p(x, y, 1);
            // const Matrix3 I3_pezT = (I3 - p * ez.transpose());
            // const Eigen::Matrix<Scalar, 2, 3> I3_pezT2 = I3_pezT.template block<2, 3>(0, 0);
            //Vector3 pk_1 = p + dt * I3_pezT * H * p;
            const Vector2 pk_1(
                    x + x*(dt*H(0, 0) - dt*H(2, 0)*x) + y*(dt*H(0, 1) - dt*H(2, 1)*x) + dt*H(0, 2) - dt*H(2, 2)*x, 
                    y + x*(dt*H(1, 0) - dt*H(2, 0)*y) + y*(dt*H(1, 1) - dt*H(2, 1)*y) + dt*H(1, 2) - dt*H(2, 2)*y);
            //pk_1(0) = x + x*(dt*H(0, 0) - dt*H(2, 0)*x) + y*(dt*H(0, 1) - dt*H(2, 1)*x) + dt*H(0, 2) - dt*H(2, 2)*x;
            //pk_1(1) = y + x*(dt*H(1, 0) - dt*H(2, 0)*y) + y*(dt*H(1, 1) - dt*H(2, 1)*y) + dt*H(1, 2) - dt*H(2, 2)*y;
            Vector3 Kinvp = mKinv * p;
            const Scalar& Kinvpx = Kinvp(0);
            const Scalar& Kinvpy = Kinvp(1);
            const Scalar& Kinvpz = Kinvp(2);
            Scalar nTKinvp = n.dot(Kinvp);
            if (InsideImage(pk_1(0), pk_1(1), imk.rows, imk.cols) && (g_use_kf == 0 || g_use_kf == 2)){
                npixel++;
                Eigen::Matrix<Scalar, 2, _D> J2;
                J2.setZero();
                // J2.template block<2, 3>(0, 3) = dt*I3_pezT2*mK*mfAlpha*nTKinvp; // vel
                // J2.template block<2, 3>(0, 12) = dt*I3_pezT2*mK*skew(Kinvp); // bw
                // J2.template block<2, 1>(0, 15) = dt*I3_pezT2*mK*mVel*nTKinvp; // alpha
                // J2.template block<2, 2>(0, 16) = dt*I3_pezT2*mK*mfAlpha*mVel*Kinvp.transpose()*N; // normal
                // Jacobian w.r.t vel
                J2(0, 3) = dt*fx*mfAlpha*nTKinvp;
                J2(0, 5) = mfAlpha*(cx*dt*nTKinvp - dt*nTKinvp*x);
                J2(1, 4) = dt*fy*mfAlpha*nTKinvp;
                J2(1, 5) = mfAlpha*(cy*dt*nTKinvp - dt*nTKinvp*y);

                // Jacobian w.r.t bw
                J2(0, 12) = -Kinvpy*(cx*dt - dt*x);
                J2(0, 13) = Kinvpx*(cx*dt - dt*x) - Kinvpz*dt*fx;
                J2(0, 14) = Kinvpy*dt*fx;
                J2(1, 12) = Kinvpz*dt*fy - Kinvpy*(cy*dt - dt*y);
                J2(1, 13) = Kinvpx*(cy*dt - dt*y);
                J2(1, 14) = -Kinvpx*dt*fy;

                // Jacobian w.r.t alpha
                J2(0, 15) = nTKinvp*(vz*(cx*dt - dt*x) + dt*fx*vx);
                J2(1, 15) = nTKinvp*(vz*(cy*dt - dt*y) + dt*fy*vy);

                // Jacobian w.r.t normal
                Eigen::Matrix<double, 2, 3> J3;
                Scalar atmp = mfAlpha*(vz*(cx*dt - dt*x) + dt*fx*vx);
                J3(0, 0) = Kinvpx*atmp;
                J3(0, 1) = Kinvpy*atmp;
                J3(0, 2) = Kinvpz*atmp;

                atmp = mfAlpha*(vz*(cy*dt - dt*y) + dt*fy*vy);
                J3(1, 0) = Kinvpx*atmp;
                J3(1, 1) = Kinvpy*atmp;
                J3(1, 2) = Kinvpz*atmp;
                J2.template block<2, 2>(0, 16) = J3*N; 
                Scalar w = g_im_weight;

                Vector2 pk_1mea(kpsk_1[i].x, kpsk_1[i].y);
                Vector2 res = (pk_1mea - pk_1.template head<2>());

                Scalar cur_chi2 = res.dot(res);
                Scalar res_norm = sqrt(cur_chi2);

                if(mbHuber){
                    if (fabs(res_norm) > g_robust_delta){
                        w *= g_robust_delta/fabs(res_norm);
                        // status[i] = 0;
                    }
                }
                imHessian += J2.transpose()*J2*w;
                JTe += J2.transpose()*res*w;
                chi2 += cur_chi2;
            }
        }

        if (mnFrameId > g_init_nkf && (g_use_kf == 1 || g_use_kf == 2)){
            for (int i = 0, iend = kpskkf.size(); i<iend; i++){
                if(!statuskf[i])
                    continue;
                Scalar x = kpskkf[i].x;
                Scalar y = kpskkf[i].y;
                Vector3 p(x, y, 1.);
                const Vector3 pk_1homo = Hkf*p;
                const Vector2 pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
                Vector3 Kinvp = mKinv * p;
                Scalar nTKinvp = n.dot(Kinvp);
                if (InsideImage(pkf(0), pkf(1), imk_1.rows, imk_1.cols) && (g_use_kf == 1 || g_use_kf == 2)){
                    npixelKF++;
                    Eigen::Matrix<Scalar, 3, _D> J3;
                    Eigen::Matrix<Scalar, 2, _D> J;
                    Eigen::Matrix<Scalar, 2, 3> J2;
                    J2.setZero();
                    J3.setZero();
                    J2(0, 0) = 1./pk_1homo(2);
                    J2(0, 2) = -pk_1homo(0)/(pk_1homo(2)*pk_1homo(2));
                    J2(1, 1) = 1./pk_1homo(2);
                    J2(1, 2) = -pk_1homo(1)/(pk_1homo(2)*pk_1homo(2));

                    J3.template block<3, 3>(0, 0) = mfAlpha*nTKinvp*mK*RKFw; // pos
                    Vector3 RwcKinvp = Rwc*Kinvp;
                    J3.template block<3, 3>(0, 6) = -mK*RKFw*skew(RwcKinvp); // att
                    J3.template block<3, 1>(0, 15) = nTKinvp*mK*posKFc; // alpha
                    J3.template block<3, 2>(0, 16) = mK*mfAlpha*posKFc*Kinvp.transpose()*N; // normal

                    Vector2 pkkfmea(mKF.mTracks[i].x, mKF.mTracks[i].y);
                    Vector2 res = pkkfmea - pkf;
                    J = J2*J3;

                    Scalar cur_chi2 = res.dot(res);
                    Scalar res_norm = sqrt(cur_chi2);
                    float w = g_im_weight_kf;

                    if(mbHuber){
                        if (fabs(res_norm) > g_robust_delta){
                            w *= g_robust_delta/fabs(res_norm);
                            // statuskf[i] = 0;
                        }
                    }
                    imHessian += J.transpose()*J*w;
                    JTe += J.transpose()*res*w;
                    chi2KF += cur_chi2;
                }
            }
        }
        LOG_END_CV_TIME_MS(tHJTe);
        // if (npixel == 0 && g_use_kf != 1)
        //     LOG(FATAL) << "npixel = 0";
        if (npixel == 0 && g_use_kf != 1){
            LOG(INFO) << "npixel = 0";
            break;
        }
        if (std::isnan(chi2))
            LOG(FATAL) << "chi2 nan";

        float avgChi2 = npixel == 0?1.:chi2/npixel;
        float avgChi2KF = npixelKF == 0?1.:chi2KF/npixelKF;
        float avgGradientKF = npixelKF == 0?0.:gradientKF/npixelKF;
        if (mbLogDebug){
            LOG(INFO) << "H = " << imHessian << std::endl << " JTe = " << JTe.transpose();
            LOG(INFO) << "iteration: " << iter << " chi2: " << chi2 << " npixel: " << npixel << " avgChi2: " << avgChi2 << " npixelKF: " << npixelKF << " chi2KF: " << chi2KF << " avgChi2KF: " << avgChi2KF << " avgGradientKF: " << avgGradientKF;
        }

        VectorState xerr;
        xerr.template segment<3>(0) = pos0 - mPos;
        xerr.template segment<3>(3) = vel0 - mVel;
        xerr.template segment<3>(6) = (rot0*mRot.inverted()).logarithmicMap();
        //LOG(INFO) << "delta R: " << rot0*mRot.inverted() << " exp(r): " << mRot.exponentialMap(xerr.template segment<3>(6))*mRot << " rot0: " << rot0;
        xerr.template segment<3>(9) = ba0 - mBa;
        xerr.template segment<3>(12) = bw0 - mBw;
        xerr(15) = alpha0 - mfAlpha;
        Vector2 nerr;
        normal0.boxMinus(mUnitDirection, nerr);
        xerr.template segment<2>(16) = nerr;

        MatrixState Jprior;
        Jprior.setIdentity();
        typename NormalVector::M2D jboxminus;
        normal0.boxMinusJac(mUnitDirection, jboxminus);
        Jprior.template block<2, 2>(16, 16) = jboxminus;
        MatrixState Pinv = (Jprior.transpose()*mP*Jprior).inverse();
        JTe += Pinv*xerr;

        MatrixState Hessian = (imHessian+Pinv);
        VectorState delta = Hessian.ldlt().solve(JTe);
        if (mbLogDebug)
        {
            LOG(INFO) << "Pinv: " << Pinv << " xerr: " << xerr.transpose() << " JT*xerr: " << xerr.transpose()*Pinv.transpose();
            LOG(INFO) << "Hessian = " << Hessian << " JTe: " << JTe.transpose();
            LOG(INFO) << "JPriorNormal = " << jboxminus;
            LOG(INFO) << "delta = " << delta.transpose();
            LOG(INFO) << "condi = " << Hessian.inverse().norm()*Hessian.norm();
            LOG(INFO) << "Pcondi =  " << mP.inverse().norm()*mP.norm();
        }

        mPos += delta.template segment<3>(0);
        mVel += delta.template segment<3>(3);
        mRot = mRot.exponentialMap(delta.template segment<3>(6))*mRot;
        mBa += delta.template segment<3>(9);
        mBw += delta.template segment<3>(12);
        mfAlpha += delta(15);
        NormalVector nout;
        mUnitDirection.boxPlus(delta.template segment<2>(16), nout);
        mUnitDirection.q_ = nout.q_;

        if (mbLogDebug){
            LOG(INFO) << "alpha = " << mfAlpha << " pos: "
                      << mPos.transpose() << " vel: " << mVel.transpose();
            LOG(INFO)  << " mRot: " << mRot;
            LOG(INFO) << "RKFc2: " << RKFw*typename NormalVector::MPD(mRot).matrix();
            LOG(INFO) << "ba = " << mBa.transpose() << " bw = " << mBw.transpose() << " norm(delta)= " << delta.norm();
        }

        if (g_draw_homography && (g_use_kf == 1 || g_use_kf == 2)){
            cv::Mat homoImg = DrawRect(imk, mKF.mImg, Hkf);
            char buf[256];
            sprintf(buf, "/tmp/homorect_%06d_%02d.png", mnFrameId, iter);
            cv::imwrite(buf, homoImg);
        }
        if(delta.norm() < 5e-2 || iter++ >= maxIter)
        {
            MatrixState A = Hessian.inverse()*imHessian;
            mP = mP - A*mP;
            mP = (mP+mP.transpose())/2.;
            if (mbLogDebug){
                LOG(INFO) << "poseterior p = " << mP;
            }

            if (g_draw_homography ){
                char buf[256];
                cv::Mat matchImg = drawMatches(imk, imk_1,
                        kpsk, kpsk_1, status);
                sprintf(buf, "/tmp/match_consecutative_%06d_%02d.png", mnFrameId, iter);
                cv::imwrite(buf, matchImg);
            }

            if (g_draw_homography && (g_use_kf == 1 || g_use_kf == 2)){
                char buf[256];
                cv::Mat matchImg = drawMatches(imk, mKF.mImg,
                        kpskkf, mKF.mTracks, statuskf);
                sprintf(buf, "/tmp/matchkf_%06d_%02d.png", mnFrameId, iter);
                cv::imwrite(buf, matchImg);
            }
            
            if (g_change_weight && (g_use_kf == 1 || g_use_kf == 2)
                    && mnFrameId == g_init_nkf) {
                g_im_weight = g_im_weight2;
                g_im_weight_kf = g_im_weight_kf2;

                if (mbLogDebug)
                    LOG(INFO) << "change weight";
            }
            if (mnFrameId <= g_init_nkf 
                   || mLastKFTracks.size() < g_kf_match_th
                   && (g_use_kf == 1 || g_use_kf == 2)
               )
            {

                mKF.mTracks.clear();
                cv::goodFeaturesToTrack(mKF.mImg, mKF.mTracks, 100, 0.1, 10);
                mLastKFTracks = mKF.mTracks;
                mKF.mFeatures = mKF.mTracks;
                mKF.mnFrameId = mnFrameId;
                mKF.mRot = mRot;
                mKF.mPos = mPos;
                mKF.mdInverseDistance = mfAlpha;
                mKF.mNormal = mUnitDirection.getVec();
                imk.copyTo(mKF.mImg);
                if (mbLogDebug)
                    LOG(INFO) << "change kf";
                // if (mnFrameId >= g_init_nkf){
                //     mvpKfs.emplace_back(new Frame(mKF));
                // }
            }
            else{
                int k = 0;
                // n = mUnitDirection.getVec();
                // Rwc = typename NormalVector::MPD(mRot).matrix();
                // RKFc = RKFw*Rwc;
                // posKFc = RKFw*mPos + PosKF;
                // Hkf = mK*(RKFc + mfAlpha*posKFc*n.transpose())*mKinv;
                for (size_t i = 0, iend = statuskf.size(); i<iend; i++){
                    // const cv::Point2f &ptk = kpskkf[i];
                    // const cv::Point2f &ptkf = mKF.mTracks[i];
                    // Vector3 pk(ptk.x, ptk.y, 1.);
                    // Vector3 pkf = Hkf*pk;
                    // pkf = pkf/pkf(2);
                    // Vector3 pkfmea(ptkf.x, ptkf.y, 1.);
                    // float res = (pkf - pkfmea).norm();
                    // LOG(INFO) << "res outlier check: " << res;
                    if(statuskf[i]){
                        mKF.mTracks[k] = mKF.mTracks[i];
                        mLastKFTracks[k++] = kpskkf[i];
                    }
                }
                if (mbLogDebug)
                    LOG(INFO) << "kf inliers: " << k;
                mKF.mTracks.resize(k);
                mLastKFTracks.resize(k);
                if (k < 10){
                    mKF.mTracks.clear();
                    cv::goodFeaturesToTrack(mKF.mImg, mKF.mTracks, 100, 0.1, 10);
                    mLastKFTracks = mKF.mTracks;
                    mKF.mFeatures = mKF.mTracks;
                    mKF.mnFrameId = mnFrameId;
                    mKF.mRot = mRot;
                    mKF.mPos = mPos;
                    mKF.mdInverseDistance = mfAlpha;
                    mKF.mNormal = mUnitDirection.getVec();
                    imk.copyTo(mKF.mImg);
                    if (mbLogDebug)
                        LOG(INFO) << "change kf";
                }
            }
            LOG_END_CV_TIME_MS(tIteration);
            break;
        }
        LOG_END_CV_TIME_MS(tIteration);
    }
    imk.copyTo(imk_1);
    Ixk.copyTo(Ixk_1);
    Iyk.copyTo(Iyk_1);
    mnFrameId++;

}

template <typename Scalar, int _D>
void State<Scalar, _D>::MeasurementUpdateKFLastMarginalization(const std::vector<IMU> &imus, const cv::Mat &imLast, const cv::Mat &imCur, Scalar dt){
    Vector3 gyro(0, 0, 0);
    for(const IMU& imu:imus)
    {
        gyro(0) += imu.gyro[0];
        gyro(1) += imu.gyro[1];
        gyro(2) += imu.gyro[2];
        if(mbLogDebug)
            LOG(INFO) << "gyro: " << imu.gyro[0] << " " << imu.gyro[1] << " " << imu.gyro[2];
    }
    gyro = 1.f/imus.size()*gyro;
    const float s = g_scale;
    if(mbLogDebug)
    {
        LOG(INFO) << "gyro = " << gyro.transpose();
        LOG(INFO) << "K = " << mK << " Kinv = " << mKinv << " dt: " << dt;
    }
    if(imk_1.empty())
    {
        cv::resize(imLast, imk_1, cv::Size(), s, s);
        if (g_use_median_blur)
            cv::medianBlur(imk_1, imk_1, g_gaussian_sigma);
        else // guassian blur
            imk_1 = HomoAlign::SmoothImage(g_gaussian_sigma, imk_1);
        if(g_k1 != 0)
        {
            cv::remap(imk_1, imk_1, remap1l_, remap2l_, cv::INTER_LINEAR);
        }
        HomoAlign::ConvertImageToFloat(imk_1);
        HomoAlign::ComputeImageDerivatives(imk_1, Ixk_1, Iyk_1);
        imk_1.copyTo(mKF.mImg);
        Ixk_1.copyTo(mKF.mIx);
        Iyk_1.copyTo(mKF.mIy);
        mKF.mnFrameId = 0;
        mnFrameId = 1;
        //mnKFId = 0;
    }

    //cv::Mat imk;
    cv::resize(imCur, imk, cv::Size(), s, s);
    if (g_use_median_blur)
        cv::medianBlur(imk, imk, g_gaussian_sigma);
    else
        imk = HomoAlign::SmoothImage(g_gaussian_sigma, imk);
    if(g_k1 != 0)
    {
        cv::remap(imk, imk, remap1l_, remap2l_, cv::INTER_LINEAR);
        //cv::imwrite("undis.png", imk);
    }
	//static int nimg = 0 ;
	//char buf[256];
	//sprintf(buf, "/tmp/%04d.png", nimg++);
	//cv::imwrite(buf, imk);


    if(mbLogDebug){
        cv::imshow("Ik", imCur);
        cv::waitKey(1);
    }


    START_CV_TIME(tSmoothImage);
    LOG_END_CV_TIME_MS(tSmoothImage);
    START_CV_TIME(tConvertImageToFloat);
    HomoAlign::ConvertImageToFloat(imk);
    LOG_END_CV_TIME_MS(tConvertImageToFloat);
    //cv::Mat Ixk, Iyk;
    START_CV_TIME(tComputeImageDerivatives);
    HomoAlign::ComputeImageDerivatives(imk, Ixk, Iyk);
    LOG_END_CV_TIME_MS(tComputeImageDerivatives);

    Vector3 ez(0, 0, 1.);
    Matrix3 I3;
    I3.setIdentity();

    int iter = 0;
    int maxIter = g_max_iteration;
    //MatrixState Pinv0 = mP.inverse();
    const float *ptrimk_1 = imk_1.ptr<float>(0, 0);
    const float *ptrIxk_1 = Ixk_1.ptr<float>(0, 0);
    const float *ptrIyk_1 = Iyk_1.ptr<float>(0, 0);

    const float *ptrimk = imk.ptr<float>(0, 0);
    const float *ptrIxk = Ixk.ptr<float>(0, 0);
    const float *ptrIyk = Iyk.ptr<float>(0, 0);

    const float *ptrimKF = mKF.mImg.ptr<float>(0, 0);
    const float *ptrIxKF = mKF.mIx.ptr<float>(0, 0);
    const float *ptrIyKF = mKF.mIy.ptr<float>(0, 0);

    Scalar alpha0 = mfAlpha;
    Vector3 ba0 = mBa;
    Vector3 bw0 = mBw;
    Vector3 pos0 = mPos;
    Vector3 vel0 = mVel;
    typename NormalVector::QPD rot0 = mRot;
    Matrix3 RKFw = typename NormalVector::MPD(mKF.mRot).matrix().transpose();
    Vector3 PosKF = -RKFw*mKF.mPos;
    if (mbLogDebug){
        LOG(INFO) << "-------------------------------------";
    }
    NormalVector normal0;
    normal0.q_ = mUnitDirection.q_;
    mA = 1.;
    mB = 0.;
    const Scalar &fx = mK(0, 0);
    const Scalar &fy = mK(1, 1);
    const Scalar &cx = mK(0, 2);
    const Scalar &cy = mK(1, 2);
    float ac = 1., bc = 0.;
    bool marginalizationC = g_marginalize_AB > 0;

    while (true) {
        START_CV_TIME(tIteration);
        Vector3 n, gyro_b;
        n = mUnitDirection.getVec();
        const Scalar &nx = n(0);
        const Scalar &ny = n(1);
        const Scalar &nz = n(2);
        const Scalar &vx = mVel(0);
        const Scalar &vy = mVel(1);
        const Scalar &vz = mVel(2);
        gyro_b = gyro - mBw;
        Eigen::Matrix<Scalar, 3, 2> N = mUnitDirection.getM();
        Matrix3 Rwc = typename NormalVector::MPD(mRot).matrix();
        Matrix3 RKFc = RKFw*Rwc;
        Vector3 posKFc = RKFw*mPos + PosKF;
        const Scalar &posKFcx = posKFc(0);
        const Scalar &posKFcy = posKFc(1);
        const Scalar &posKFcz = posKFc(2);
        Matrix3 Hkf = mK*(RKFc+mfAlpha*posKFc*n.transpose())*mKinv;
        LOG(INFO) << "RKFw: " << RKFw << " RKFc: " << RKFc << " pos: " << mPos.transpose() << " n: " << n.transpose() << " mRot: " << mRot << " alpha: " << mfAlpha << " poseKFc: " << posKFc;

        MatrixState imHessian;
        imHessian.setZero();
        VectorState JTe;
        JTe.setZero();

        // for schur complementary 
        Matrix2 HD, HDC;// C for continuous
        Eigen::Matrix<double, _D, 2> HB, HBC; // C for continuous
        HD.setZero();
        HB.setZero();
        HDC.setZero();
        HBC.setZero();
        Vector2 JTeAB, JTeABC; // C for continuous
        JTeAB.setZero();
        JTeABC.setZero();

        Scalar chi2 = 0.f;
        Scalar chi2KF = 0.f;
        int border = 3;
        int step = 1;

        int npixel = 0;
        int npixelKF = 0;
        Matrix3 H = mK*(skew(gyro_b) + mfAlpha*mVel*n.transpose())*mKinv;

        START_CV_TIME(tHJTe);
        float gradientKF = 0.;
        for(int y = border; y<imk.rows-border; y+=step)
            for(int x = border; x<imk.cols-border; x+=step ) {
                const Vector3 p(x, y, 1);
                // const Matrix3 I3_pezT = (I3 - p * ez.transpose());
                // const Eigen::Matrix<Scalar, 2, 3> I3_pezT2 = I3_pezT.template block<2, 3>(0, 0);
                //Vector3 pk_1 = p + dt * I3_pezT * H * p;
                const Vector2 pk_1(
                        x + x*(dt*H(0, 0) - dt*H(2, 0)*x) + y*(dt*H(0, 1) - dt*H(2, 1)*x) + dt*H(0, 2) - dt*H(2, 2)*x, 
                        y + x*(dt*H(1, 0) - dt*H(2, 0)*y) + y*(dt*H(1, 1) - dt*H(2, 1)*y) + dt*H(1, 2) - dt*H(2, 2)*y);
                //pk_1(0) = x + x*(dt*H(0, 0) - dt*H(2, 0)*x) + y*(dt*H(0, 1) - dt*H(2, 1)*x) + dt*H(0, 2) - dt*H(2, 2)*x;
                //pk_1(1) = y + x*(dt*H(1, 0) - dt*H(2, 0)*y) + y*(dt*H(1, 1) - dt*H(2, 1)*y) + dt*H(1, 2) - dt*H(2, 2)*y;
                //if ((pk_1(0)) < 0 || (pk_1(0)) >= imk_1.cols-1
                //    || (pk_1(1)) < 0 || (pk_1(1)) >= imk_1.rows-1)
                //    continue;
                Vector3 Kinvp = mKinv * p;
                const Scalar& Kinvpx = Kinvp(0);
                const Scalar& Kinvpy = Kinvp(1);
                const Scalar& Kinvpz = Kinvp(2);
                Scalar nTKinvp = n.dot(Kinvp);
                if (InsideImage(pk_1(0), pk_1(1), imk.rows, imk.cols) && (g_use_kf == 0 || g_use_kf == 2)){
                    npixel++;
                    Eigen::Matrix<Scalar, 1, 2> J1;
                    Eigen::Matrix<Scalar, 1, _D> J;
                    Eigen::Matrix<Scalar, 2, _D> J2;
                    J2.setZero();
                    // J2.template block<2, 3>(0, 3) = dt*I3_pezT2*mK*mfAlpha*nTKinvp; // vel
                    // J2.template block<2, 3>(0, 12) = dt*I3_pezT2*mK*skew(Kinvp); // bw
                    // J2.template block<2, 1>(0, 15) = dt*I3_pezT2*mK*mVel*nTKinvp; // alpha
                    // J2.template block<2, 2>(0, 16) = dt*I3_pezT2*mK*mfAlpha*mVel*Kinvp.transpose()*N; // normal

                    // Jacobian w.r.t vel
                    J2(0, 3) = dt*fx*mfAlpha*nTKinvp;
                    J2(0, 5) = mfAlpha*(cx*dt*nTKinvp - dt*nTKinvp*x);
                    J2(1, 4) = dt*fy*mfAlpha*nTKinvp;
                    J2(1, 5) = mfAlpha*(cy*dt*nTKinvp - dt*nTKinvp*y);

                    // Jacobian w.r.t bw
                    J2(0, 12) = -Kinvpy*(cx*dt - dt*x);
                    J2(0, 13) = Kinvpx*(cx*dt - dt*x) - Kinvpz*dt*fx;
                    J2(0, 14) = Kinvpy*dt*fx;
                    J2(1, 12) = Kinvpz*dt*fy - Kinvpy*(cy*dt - dt*y);
                    J2(1, 13) = Kinvpx*(cy*dt - dt*y);
                    J2(1, 14) = -Kinvpx*dt*fy;

                    // Jacobian w.r.t alpha
                    J2(0, 15) = nTKinvp*(vz*(cx*dt - dt*x) + dt*fx*vx);
                    J2(1, 15) = nTKinvp*(vz*(cy*dt - dt*y) + dt*fy*vy);

                    // Jacobian w.r.t normal
                    Eigen::Matrix<double, 2, 3> J3;
                    Scalar atmp = mfAlpha*(vz*(cx*dt - dt*x) + dt*fx*vx);
                    J3(0, 0) = Kinvpx*atmp;
                    J3(0, 1) = Kinvpy*atmp;
                    J3(0, 2) = Kinvpz*atmp;

                    atmp = mfAlpha*(vz*(cy*dt - dt*y) + dt*fy*vy);
                    J3(1, 0) = Kinvpx*atmp;
                    J3(1, 1) = Kinvpy*atmp;
                    J3(1, 2) = Kinvpz*atmp;
                    J2.template block<2, 2>(0, 16) = J3*N; 

                    int last_u_i = static_cast<int>(floor(pk_1(0)));
                    int last_v_i = static_cast<int>(floor(pk_1(1)));
                    const float subpix_u_ref = pk_1(0) - last_u_i;
                    const float subpix_v_ref = pk_1(1)- last_v_i;
                    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                    const float w_ref_br = subpix_u_ref * subpix_v_ref;
                    int cur_u = x;
                    int cur_v = y;
                    const int step = imk_1.cols;
                    int last_idx = last_u_i + last_v_i*step;
                    int cur_idx = cur_u + cur_v*step;
                    const float *ptr_Ixk_1 = ptrIxk_1 + last_idx;
                    float last_gx = w_ref_tl * VALUE_FROM_ADDRESS(ptr_Ixk_1) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_Ixk_1+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_Ixk_1+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_Ixk_1+step+1);
                    //J1(0) = (VALUE_FROM_ADDRESS(ptrIxk+cur_idx) + last_gx) / 2.f;
                    J1(0) = last_gx;

                    const float *ptr_Iyk_1 = ptrIyk_1 + last_idx;
                    float last_gy = w_ref_tl * VALUE_FROM_ADDRESS(ptr_Iyk_1) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_Iyk_1+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_Iyk_1+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_Iyk_1+step+1);
                    //J1(1) = (VALUE_FROM_ADDRESS(ptrIyk+cur_idx) + last_gy) / 2.f;
                    J1(1) = last_gy;
                    J = ac * J1 * J2;
                    //LOG(INFO) << "J = " << J << " J1: " << J1 << " J2: " << J2 << " J3: " << J3;
                    //LOG(INFO) << "J = " << J;
                    float w = g_im_weight*J1.norm();
                    //float w = g_im_weight;
                    //LOG(INFO) << "w = " << w;

                    const float *ptr_imk_1 = ptrimk_1 + last_idx;
                    float last_intensity = w_ref_tl * VALUE_FROM_ADDRESS(ptr_imk_1) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_imk_1+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_imk_1+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_imk_1+step+1);
                    float res = (VALUE_FROM_ADDRESS(ptrimk+cur_idx) - ac * last_intensity - bc);
                    //LOG(INFO) << "J1 = " << J1 << " curp: " << VALUE_FROM_ADDRESS(ptrimk+cur_idx) << " lastpix: " << last_intensity;

                    if(mbHuber){
                        if (fabs(res) > g_robust_delta)
                            w *= g_robust_delta/fabs(res);
                    }
                    imHessian += J.transpose() * J * w;

                    JTe += res * J.transpose() * w;
                    if (marginalizationC){
                        Eigen::Matrix<Scalar, 1, 2> JAB;
                        JAB(0) = last_intensity;
                        JAB(1) = 1.;
                        HDC += JAB.transpose() * JAB * w;
                        HBC += J.transpose() * JAB * w;
                        JTeABC += res * JAB * w;
                    }
                    chi2 += res * res;

                }

                if (mnFrameId <= g_init_nkf)
                    continue;
                
                const Vector3 pk_1homo = Hkf*p;
                const Vector2 pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
                if (InsideImage(pkf(0), pkf(1), imk_1.rows, imk_1.cols) && (g_use_kf == 1 || g_use_kf == 2)){
                    npixelKF++;
                    Eigen::Matrix<Scalar, 1, 2> J1;
                    Eigen::Matrix<Scalar, 3, _D> J3;
                    Eigen::Matrix<Scalar, 1, _D> J;
                    Eigen::Matrix<Scalar, 2, 3> J2;
                    J2.setZero();
                    J3.setZero();
                    J2(0, 0) = 1./pk_1homo(2);
                    J2(0, 2) = -pk_1homo(0)/(pk_1homo(2)*pk_1homo(2));
                    J2(1, 1) = 1./pk_1homo(2);
                    J2(1, 2) = -pk_1homo(1)/(pk_1homo(2)*pk_1homo(2));

                    J3.template block<3, 3>(0, 0) = mfAlpha*nTKinvp*mK*RKFw; // pos
                    Vector3 RwcKinvp = Rwc*Kinvp;
                    J3.template block<3, 3>(0, 6) = -mK*RKFw*skew(RwcKinvp); // att
                    // J3.template block<3, 1>(0, 15) = nTKinvp*mK*posKFc; // alpha
                    J3(0, 15) = cx*nTKinvp*posKFcz + fx*nTKinvp*posKFcx;
                    J3(1, 15) = cy*nTKinvp*posKFcz + fy*nTKinvp*posKFcy;
                    J3(2, 15) = nTKinvp*posKFcz;

                    // J3.template block<3, 2>(0, 16) = mK*mfAlpha*posKFc*Kinvp.transpose()*N; // normal
                    Matrix3 J4;
                    Scalar atmp = (cx*mfAlpha*posKFcz + fx*mfAlpha*posKFcx);
                    J4(0, 0) = Kinvpx*atmp;
                    J4(0, 1) = Kinvpy*atmp;
                    J4(0, 2) = Kinvpz*atmp;

                    atmp = (cy*mfAlpha*posKFcz + fy*mfAlpha*posKFcy);
                    J4(1, 0) = Kinvpx*atmp;
                    J4(1, 1) = Kinvpy*atmp;
                    J4(1, 2) = Kinvpz*atmp;

                    atmp = mfAlpha*posKFcz;
                    J4(2, 0) = Kinvpx*atmp;
                    J4(2, 1) = Kinvpy*atmp;
                    J4(2, 2) = Kinvpz*atmp;

                    J3.template block<3, 2>(0, 16) = J4*N;

#if 1
                    int last_u_i = static_cast<int>(floor(pkf(0)));
                    int last_v_i = static_cast<int>(floor(pkf(1)));
                    const float subpix_u_ref = pkf(0) - last_u_i;
                    const float subpix_v_ref = pkf(1) - last_v_i;
                    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                    const float w_ref_br = subpix_u_ref * subpix_v_ref;
                    int cur_u = x;
                    int cur_v = y;
                    const int step = mKF.mImg.cols;
                    int last_idx = last_u_i + last_v_i*step;
                    int cur_idx = cur_u + cur_v*step;
                    const float *ptr_IxKF = ptrIxKF + last_idx;
                    float last_gx = w_ref_tl * VALUE_FROM_ADDRESS(ptr_IxKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_IxKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_IxKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_IxKF+step+1);
                    //J1(0) = (VALUE_FROM_ADDRESS(ptrIxk+cur_idx) + last_gx) / 2.f;
                    J1(0) = last_gx;

                    const float *ptr_IyKF = ptrIyKF + last_idx;
                    float last_gy = w_ref_tl * VALUE_FROM_ADDRESS(ptr_IyKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_IyKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_IyKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_IyKF+step+1);
                    //J1(1) = (VALUE_FROM_ADDRESS(ptrIyk+cur_idx) + last_gy) / 2.f;
                    J1(1) = last_gy;
                    //LOG(INFO) << J1.transpose() << " aver: " << (VALUE_FROM_ADDRESS(ptrIxk+cur_idx) + last_gx) / 2.f << " " << (VALUE_FROM_ADDRESS(ptrIyk+cur_idx) + last_gy) / 2.f;
                    
                    J = mA * J1 * J2 * J3;
                    //LOG(INFO) << "J = " << J << " J1: " << J1 << " J2: " << J2 << " J3: " << J3;
                    gradientKF += J1.norm();
                    float w = g_im_weight_kf*J1.norm();
                    //float w = g_im_weight;
                    //LOG(INFO) << "w = " << w;

                    const float *ptr_imKF = ptrimKF + last_idx;
                    float last_intensity = w_ref_tl * VALUE_FROM_ADDRESS(ptr_imKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_imKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_imKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_imKF+step+1);
                    float res = (VALUE_FROM_ADDRESS(ptrimk+cur_idx) - mA*last_intensity - mB);
                    //LOG(INFO) << "J = " << J;

                    if(mbHuber){
                        if (fabs(res) > g_robust_delta)
                            w *= g_robust_delta/fabs(res);
                    }
                    imHessian += J.transpose() * J * w;

                    Eigen::Matrix<Scalar, 1, 2> JAB;
                    JAB(0) = last_intensity;
                    JAB(1) = 1.;
                    HD += JAB.transpose() * JAB * w;
                    HB += J.transpose() * JAB * w;
#else
                    int last_u_i = cvRound(pkf.at<float>(0));
                    int last_v_i = cvRound(pkf.at<float>(1));
                    int cur_u = x;
                    int cur_v = y;

                    J1.at<float>(0) =(Ixk.at<float>(cur_v, cur_u) + Ixk_1.at<float>(last_v_i, last_u_i)) / 2.f;
                    J1.at<float>(1) =(Iyk.at<float>(cur_v, cur_u) + Iyk_1.at<float>(last_v_i, last_u_i)) / 2.f;
                    J = J1*J2;
                    const float w = 5;
                    imHessian += J.t() * J * w;
                    float res = (imk.at<float>(cur_v, cur_u) - imk_1.at<float>(last_v_i, last_u_i));
#endif
                    JTe += res * J.transpose() * w;
                    JTeAB += res * JAB.transpose() * w;
                    chi2KF += res * res;
                }
            }
        LOG_END_CV_TIME_MS(tHJTe);
        if (npixel == 0 && g_use_kf != 1)
            LOG(FATAL) << "npixel = 0";
        else 
        if (marginalizationC){
            HDC.diagonal().array() += g_weight_opt_cf_ab;
            Matrix2 HDCinv = HDC.inverse();
            MatrixState Hschur = HBC*HDCinv*HBC.transpose();
            VectorState bschur = HBC*HDCinv*JTeABC;
            imHessian -= Hschur;
            JTe -= bschur;
        }
        if (std::isnan(chi2))
            LOG(FATAL) << "chi2 nan";
        if ((g_use_kf == 1 || g_use_kf == 2) && g_marginalize_AB > 0 && mnFrameId > g_init_nkf){
            if(npixelKF == 0){
                // LOG(FATAL) << "npixelkf = 0";
            }
            else{
                Matrix2 HDinv = HD.inverse();
                MatrixState Hschur = HB*HDinv*HB.transpose();
                VectorState bschur = HB*HDinv*JTeAB;
                imHessian -= Hschur;
                JTe -= bschur;
                if (mbLogDebug){
                    LOG(INFO) << "Hschur: " << Hschur;
                    LOG(INFO) << "bschur: " << bschur.transpose();
                    LOG(INFO) << "HD: " << HD;
                    LOG(INFO) << "HB: " << HB;
                    LOG(INFO) << "HDinv: " << HDinv;
                }

            }
        }
        

        float avgChi2 = npixel == 0?1.:chi2/npixel;
        float avgChi2KF = npixelKF == 0?1.:chi2KF/npixelKF;
        float avgGradientKF = npixelKF == 0?0.:gradientKF/npixelKF;
        // LOG(INFO) << "npixel: " << npixel;
        if (mbLogDebug){
            LOG(INFO) << "H = " << imHessian << std::endl << " JTe = " << JTe.transpose();
            LOG(INFO) << "iter: " << iter << " chi2: " << chi2 << " npixel: " << npixel << " avgChi2: " << avgChi2 << " npixelKF: " << npixelKF << " chi2KF: " << chi2KF << " avgChi2KF: " << avgChi2KF << " avgGradientKF: " << avgGradientKF;
        }

        VectorState xerr;
        xerr.template segment<3>(0) = pos0 - mPos;
        xerr.template segment<3>(3) = vel0 - mVel;
        xerr.template segment<3>(6) = (rot0*mRot.inverted()).logarithmicMap();
        //LOG(INFO) << "delta R: " << rot0*mRot.inverted() << " exp(r): " << mRot.exponentialMap(xerr.template segment<3>(6))*mRot << " rot0: " << rot0;
        xerr.template segment<3>(9) = ba0 - mBa;
        xerr.template segment<3>(12) = bw0 - mBw;
        xerr(15) = alpha0 - mfAlpha;
        Vector2 nerr;
        normal0.boxMinus(mUnitDirection, nerr);
        xerr.template segment<2>(16) = nerr;

        MatrixState Jprior;
        Jprior.setIdentity();
        typename NormalVector::M2D jboxminus;
        normal0.boxMinusJac(mUnitDirection, jboxminus);
        Jprior.template block<2, 2>(16, 16) = jboxminus;
        MatrixState Pinv = (Jprior.transpose()*mP*Jprior).inverse();
        JTe += Pinv*xerr;

        MatrixState Hessian = (imHessian+Pinv);
        VectorState delta = Hessian.ldlt().solve(JTe);
        if (mbLogDebug)
        {
            LOG(INFO) << "Pinv: " << Pinv << " xerr: " << xerr.transpose() << " JT*xerr: " << xerr.transpose()*Pinv.transpose();
            LOG(INFO) << "Hessian = " << Hessian << " JTe: " << JTe.transpose();
            LOG(INFO) << "delta = " << delta.transpose();
            LOG(INFO) << "condi = " << Hessian.inverse().norm()*Hessian.norm();
            LOG(INFO) << "Pcondi =  " << mP.inverse().norm()*mP.norm();
        }

        mPos += delta.template segment<3>(0);
        mVel += delta.template segment<3>(3);
        mRot = mRot.exponentialMap(delta.template segment<3>(6))*mRot;
        mBa += delta.template segment<3>(9);
        mBw += delta.template segment<3>(12);
        mfAlpha += delta(15);
        NormalVector nout;
        mUnitDirection.boxPlus(delta.template segment<2>(16), nout);
        mUnitDirection.q_ = nout.q_;

        if ((g_use_kf == 1 || g_use_kf == 2) && g_marginalize_AB > 0 && mnFrameId > g_init_nkf && npixelKF != 0){
            Matrix2 HDinv = HD.inverse();
            Vector2 deltaAB = HDinv*(JTeAB - HB.transpose()*delta);
            mA += deltaAB(0);
            mB += deltaAB(1);
            if (mbLogDebug){
                LOG(INFO) << "marginalized mA: " << mA << " mB: " << mB; 
            }
        }
        if (g_use_kf != 1 && npixel > 0 && marginalizationC){
            Matrix2 HDCinv = HDC.inverse();
            Vector2 deltaAB = HDCinv*(JTeABC - HBC.transpose()*delta);
            ac += deltaAB(0);
            bc += deltaAB(1);
            LOG(INFO) << "Ac: " << ac << " Bc: " << bc;
        }

        if (mbLogDebug){
            LOG(INFO) << "alpha = " << mfAlpha << " pos: "
                      << mPos.transpose() << " vel: " << mVel.transpose();
            LOG(INFO)  << " mRot: " << mRot;
            LOG(INFO) << "RKFc2: " << RKFw*typename NormalVector::MPD(mRot).matrix();
            LOG(INFO) << "ba = " << mBa.transpose() << " bw = " << mBw.transpose() << " norm(delta)= " << delta.norm();
        }

        if (g_draw_homography && (g_use_kf == 1 || g_use_kf == 2)){
            cv::Mat homoImg = DrawRect(imk, mKF.mImg, Hkf);
            Scalar gradient = GradientMagRMS(Ixk, Iyk);
            char buf[256];
            sprintf(buf, "gra: %.2f", avgGradientKF);
            cv::putText(homoImg, buf, cv::Point(5, 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

            sprintf(buf, "kfid: %d", mKF.mnFrameId);
            cv::putText(homoImg, buf, cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

            sprintf(buf, "chi2: %.2f", avgChi2KF);
            cv::putText(homoImg, buf, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));
            sprintf(buf, "/tmp/homorect_%06d_%02d.png", mnFrameId, iter);
            cv::imwrite(buf, homoImg);
            // LOG(INFO) << std::string(buf);
        }
        if(delta.norm() < 5e-2 || iter++ >= maxIter)
        {
            MatrixState A = Hessian.inverse()*imHessian;
            mP = mP - A*mP;
            mP = (mP+mP.transpose())/2.;
            if (mbLogDebug){
                LOG(INFO) << "poseterior p = " << mP;
            }
            
            std::vector<cv::Point2f> srcPts, dstPts;
            HomograpyMapImagePoints(Hkf,  srcPts, dstPts, imk.rows, imk.cols);
            float iou = PolygonIntersection::IOU(srcPts, dstPts);
            mLastIOU = iou;
            if (g_change_weight && (g_use_kf == 1 || g_use_kf == 2)
                    && mnFrameId == g_init_nkf) {
                g_im_weight = g_im_weight2;
                g_im_weight_kf = g_im_weight_kf2;
                LOG(INFO) << "change weight";

            }
            if (mnFrameId < g_init_nkf 
                   || iou < g_kf_coverage 
                   || avgGradientKF < g_avg_kf_grad_th
                   && (g_use_kf == 1 || g_use_kf == 2)
                   // || avgChi2 > 0.1
                    )
            {
                int kfidx = g_search_kf > 0?SearchOverlapKF():-1;
                if (kfidx < 0){
                    mKF.mnFrameId = mnFrameId;
                    imk.copyTo(mKF.mImg);
                    Ixk.copyTo(mKF.mIx);
                    Iyk.copyTo(mKF.mIy);
                    mKF.mRot = mRot;
                    mKF.mPos = mPos;
                    mKF.mdInverseDistance = mfAlpha;
                    mKF.mNormal = mUnitDirection.getVec();
                    LOG(INFO) << "change kf: " << mnFrameId << " iou: " << iou << " last kfid: " << mnLastKFIdx;

                    //std::vector<Frame*>::iterator itLastKF = std::find_if(mvpKfs.begin(), mvpKfs.end(), [&](const Frame* pkf){
                    //        return pkf->mnFrameId == mKF.mnFrameId;
                    //        });
                    if (mnFrameId >= g_init_nkf){
                        // create current frame as kf
                        // assign the last kf id
                        Frame* pKFCur = new Frame(mKF);
                        mvpKfs.emplace_back(pKFCur);
                        if (mnLastKFIdx > -1){
                            Scalar yaw;
                            Vector3 r12;
                            Frame* pKF1 = mvpKfs[mnLastKFIdx];
                            YawPosMeasurement(pKF1->mRot, pKFCur->mRot, pKF1->mPos, pKFCur->mPos, yaw, r12);
                            mvMeasurement.emplace_back(mvpKfs.size() - 1, mnLastKFIdx, yaw, r12);
                            LOG(INFO) << "pair: 1) " << mvpKfs.size() - 1 << " 2) " << mnLastKFIdx;
                        }
                        mnLastKFIdx = mvpKfs.size() - 1;
                    }
                }else{
                    mKF = *mvpKfs[kfidx];
                    mnLastKFIdx = kfidx;
                    LOG(INFO) << "change kf to previous kf: " << mKF.mnFrameId << " iou: " << iou;
                }
            }
            LOG_END_CV_TIME_MS(tIteration);
            break;
        }
        LOG_END_CV_TIME_MS(tIteration);
    }
    imk.copyTo(imk_1);
    Ixk.copyTo(Ixk_1);
    Iyk.copyTo(Iyk_1);
    //imk_1 = imk.clone();
    //Ixk_1 = Ixk.clone();
    //Iyk_1 = Iyk.clone();
    mnFrameId++;

}

template <typename Scalar, int _D>
void State<Scalar, _D>::YawPosMeasurement(const typename NormalVector::QPD &q1, const typename NormalVector::QPD &q2, const Vector3 &p1, const Vector3 &p2, Scalar &yaw, Vector3 &r12){
    Matrix3 Rwck = typename NormalVector::MPD(q2).matrix();
    Matrix3 Rwck_1 = typename NormalVector::MPD(q1).matrix();
    Matrix3 Rcwk = Rwck.transpose();
    Matrix3 Rcwk_1 = Rwck_1.transpose();

    Matrix3 Rpsik, Rnk, Rpsik_1, Rnk_1, RnkT, Rnk_1T;
    RotationDecomposition(Rcwk, Rnk, Rpsik);
    // LOG(INFO) << "Rpsik: " << Rpsik;
    RnkT = Rnk.transpose();
    Scalar psik = atan2(-Rpsik(1, 0), Rpsik(0, 0));

    RotationDecomposition(Rcwk_1, Rnk_1, Rpsik_1);
    Rnk_1T = Rnk_1.transpose();
    Scalar psik_1 = atan2(-Rpsik_1(1, 0), Rpsik_1(0, 0));
    yaw = psik_1 - psik;
    r12 = p2 - p1;
}

template <typename Scalar, int _D>
float State<Scalar, _D>::OptimizeCF(Frame* pKFLoop, Frame* pKFCur, float scale){
    const Vector3 &n = pKFCur->mNormal;
    const Vector3 &n2 = pKFLoop->mNormal;
    int iter = 0;
    int maxIter = g_optimize_kf_max_iter;
    // float scale = 1.;
    Matrix3 K; 
    K.setIdentity();
    K.template block<2, 3>(0, 0) = mK.template block<2, 3>(0, 0)*scale;
    // LOG(INFO) << "K: " << K;

    Matrix3 Kinv = K.inverse();
    cv::Mat ICur, IxCur, IyCur;
    cv::Mat ILoop, IxLoop, IyLoop;
    if (scale < 1.){
        cv::resize(pKFCur->mImg, ICur, cv::Size(), scale, scale);
        cv::resize(pKFCur->mIx, IxCur, cv::Size(), scale, scale);
        cv::resize(pKFCur->mIy, IyCur, cv::Size(), scale, scale);

        cv::resize(pKFLoop->mImg, ILoop, cv::Size(), scale, scale);
        cv::resize(pKFLoop->mIx, IxLoop, cv::Size(), scale, scale);
        cv::resize(pKFLoop->mIy, IyLoop, cv::Size(), scale, scale);
    }else{
        ICur = pKFCur->mImg.clone();
        IxCur = pKFCur->mIx.clone();
        IyCur = pKFCur->mIy.clone();

        ILoop = pKFLoop->mImg.clone();
        IxLoop = pKFLoop->mIx.clone();
        IyLoop = pKFLoop->mIy.clone();
    }
    // const cv::Mat &ICur = pKFCur->mImg;
    // const cv::Mat &IxCur = pKFCur->mIx;
    // const cv::Mat &IyCur = pKFCur->mIy;
    // const cv::Mat &ILoop = pKFLoop->mImg;
    // const cv::Mat &IxLoop = pKFLoop->mIx;
    // const cv::Mat &IyLoop = pKFLoop->mIy;
    const float *ptrimk = ICur.ptr<float>(0, 0);
    const float *ptrIxk = IxCur.ptr<float>(0, 0);
    const float *ptrIyk = IyCur.ptr<float>(0, 0);

    const float *ptrimKF = ILoop.ptr<float>(0, 0);
    const float *ptrIxKF = IxLoop.ptr<float>(0, 0);
    const float *ptrIyKF = IyLoop.ptr<float>(0, 0);
    float A = 1.;
    float B = 0.;
    bool use_gn = true;
    float lambda = use_gn?0:5e3;
    Vector3 ez(0, 0, 1.);
    Matrix3 skewez = skew(ez);
    Matrix3 RwKF = typename NormalVector::MPD(pKFLoop->mRot).matrix();
    Matrix3 RKFw = RwKF.transpose();
    Matrix3 Rwc = typename NormalVector::MPD(pKFCur->mRot).matrix();
    Matrix3 Rcw = Rwc.transpose();
    Matrix3 Rpsi, Rn;
    RotationDecomposition(Rcw, Rn, Rpsi);
    Scalar psi = atan2(-Rpsi(1, 0), Rpsi(0, 0));

    const int border = 3;
    const int step = ICur.cols;
    int mu = 2;
    int innerMaxIter = 5;

    while (true) {
        //START_CV_TIME(tIterationOptimizeCF);
        // Rpsi = RotZT(psi);
        Rpsi = RotZ(psi);
        Rcw = Rn*Rpsi;
        Rwc = Rcw.transpose();
        Matrix3 RKFc = RKFw*Rwc;
        Vector3 posKFc = RKFw*(pKFCur->mPos - pKFLoop->mPos);
        Matrix3 Hkf = K*(RKFc + pKFCur->mdInverseDistance*posKFc*n.transpose())*Kinv;

        Matrix3 RcKF = RKFc.transpose();
        Vector3 wrcKF = pKFLoop->mPos - pKFCur->mPos;
        Vector3 poscKF = Rcw*wrcKF;
        Matrix3 Hkf2 = K*(RcKF + pKFLoop->mdInverseDistance * poscKF * n2.transpose())*Kinv;
        //LOG(INFO) << "RKFw: " << RKFw << " RKFc: " << RKFc << " pos: " << mPos.transpose() << " n: " << n.transpose() << " mRot: " << mRot << " alpha: " << mfAlpha << " poseKFc: " << posKFc;

        Matrix6 imHessian;
        imHessian.setZero();
        Vector6 JTe;
        JTe.setZero();
        Scalar chi2KF = 0.f;

        int npixelKF = 0;
        for(int y = border; y<ICur.rows-border; y++)
            for(int x = border; x<ICur.cols-border; x++) {
                const Vector3 p(x, y, 1);
                Vector3 Kinvp = Kinv * p;
                Scalar nTKinvp = n.dot(Kinvp);
                Vector3 pk_1homo = Hkf*p;
                Vector2 pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
                if (InsideImage(pkf(0), pkf(1), ICur.rows, ICur.cols)){
                    npixelKF++;
                    Eigen::Matrix<Scalar, 1, 2> J1;
                    Eigen::Matrix<Scalar, 3, 6> J3;
                    Eigen::Matrix<Scalar, 1, 6> J;
                    Eigen::Matrix<Scalar, 2, 3> J2;
                    J2.setZero();
                    J3.setZero();
                    J2(0, 0) = 1./pk_1homo(2);
                    J2(0, 2) = -pk_1homo(0)/(pk_1homo(2)*pk_1homo(2));
                    J2(1, 1) = 1./pk_1homo(2);
                    J2(1, 2) = -pk_1homo(1)/(pk_1homo(2)*pk_1homo(2));

                    J3.template block<3, 3>(0, 0) = pKFCur->mdInverseDistance*nTKinvp*K*RKFw; // pos
                    J3.template block<3, 1>(0, 3) = K*RKFw*skewez*Rwc*Kinvp; // psi
                    int last_u_i = static_cast<int>(floor(pkf(0)));
                    int last_v_i = static_cast<int>(floor(pkf(1)));
                    const float subpix_u_ref = pkf(0) - last_u_i;
                    const float subpix_v_ref = pkf(1) - last_v_i;
                    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                    const float w_ref_br = subpix_u_ref * subpix_v_ref;
                    int cur_u = x;
                    int cur_v = y;
                    int last_idx = last_u_i + last_v_i*step;
                    int cur_idx = cur_u + cur_v*step;
                    const float *ptr_IxKF = ptrIxKF + last_idx;
                    float last_gx = w_ref_tl * VALUE_FROM_ADDRESS(ptr_IxKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_IxKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_IxKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_IxKF+step+1);
                    //J1(0) = (VALUE_FROM_ADDRESS(ptrIxk+cur_idx) + last_gx) / 2.f;
                    J1(0) = last_gx;

                    const float *ptr_IyKF = ptrIyKF + last_idx;
                    float last_gy = w_ref_tl * VALUE_FROM_ADDRESS(ptr_IyKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_IyKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_IyKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_IyKF+step+1);
                    //J1(1) = (VALUE_FROM_ADDRESS(ptrIyk+cur_idx) + last_gy) / 2.f;
                    J1(1) = last_gy;
                    //LOG(INFO) << J1.transpose() << " aver: " << (VALUE_FROM_ADDRESS(ptrIxk+cur_idx) + last_gx) / 2.f << " " << (VALUE_FROM_ADDRESS(ptrIyk+cur_idx) + last_gy) / 2.f;
                    
                    J = A * J1 * J2 * J3;
                    //LOG(INFO) << "J = " << J << " J1: " << J1 << " J2: " << J2 << " J3: " << J3;
                    float w = J1.norm();
                    //float w = g_im_weight;
                    //LOG(INFO) << "w = " << w;

                    const float *ptr_imKF = ptrimKF + last_idx;
                    float last_intensity = w_ref_tl * VALUE_FROM_ADDRESS(ptr_imKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_imKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_imKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_imKF+step+1);
                    float res = (VALUE_FROM_ADDRESS(ptrimk+cur_idx) - A*last_intensity - B);
                    //LOG(INFO) << "J = " << J;

                    if (fabs(res) > g_robust_delta)
                        w *= g_robust_delta/fabs(res);
                    J(0, 4) = last_intensity;
                    J(0, 5) = 1.;
                    imHessian += J.transpose() * J * w;
                    JTe += res * J.transpose() * w;
                    chi2KF += res * res;
                }

                nTKinvp = n2.dot(Kinvp);
                pk_1homo = Hkf2*p;
                pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
                if (InsideImage(pkf(0), pkf(1), ICur.rows, ICur.cols)){
                    npixelKF++;
                    Eigen::Matrix<Scalar, 1, 2> J1;
                    Eigen::Matrix<Scalar, 3, 6> J3;
                    Eigen::Matrix<Scalar, 1, 6> J;
                    Eigen::Matrix<Scalar, 2, 3> J2;
                    J2.setZero();
                    J3.setZero();
                    J2(0, 0) = 1./pk_1homo(2);
                    J2(0, 2) = -pk_1homo(0)/(pk_1homo(2)*pk_1homo(2));
                    J2(1, 1) = 1./pk_1homo(2);
                    J2(1, 2) = -pk_1homo(1)/(pk_1homo(2)*pk_1homo(2));

                    J3.template block<3, 3>(0, 0) = -pKFLoop->mdInverseDistance*nTKinvp*K*Rcw; // pos
                    J3.template block<3, 1>(0, 3) = -K*Rn*skewez*Rpsi*RwKF*Kinvp - pKFLoop->mdInverseDistance*nTKinvp*K*Rn*skewez*Rpsi*wrcKF; // psi
                    int last_u_i = static_cast<int>(floor(pkf(0)));
                    int last_v_i = static_cast<int>(floor(pkf(1)));
                    const float subpix_u_ref = pkf(0) - last_u_i;
                    const float subpix_v_ref = pkf(1) - last_v_i;
                    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                    const float w_ref_br = subpix_u_ref * subpix_v_ref;
                    int cur_u = x;
                    int cur_v = y;
                    int last_idx = last_u_i + last_v_i*step;
                    int cur_idx = cur_u + cur_v*step;

                    const float *ptr_IxKF = ptrIxk + last_idx;
                    float last_gx = w_ref_tl * VALUE_FROM_ADDRESS(ptr_IxKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_IxKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_IxKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_IxKF+step+1);
                    J1(0) = last_gx;

                    const float *ptr_IyKF = ptrIyk + last_idx;
                    float last_gy = w_ref_tl * VALUE_FROM_ADDRESS(ptr_IyKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_IyKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_IyKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_IyKF+step+1);
                    J1(1) = last_gy;
                    
                    J = J1 * J2 * J3;
                    //LOG(INFO) << "J = " << J << " J1: " << J1 << " J2: " << J2 << " J3: " << J3;
                    float w = J1.norm();
                    //float w = g_im_weight;
                    //LOG(INFO) << "w = " << w;

                    const float *ptr_imKF = ptrimk + last_idx;
                    float last_intensity = 
                        w_ref_tl * VALUE_FROM_ADDRESS(ptr_imKF) +
                        w_ref_tr * VALUE_FROM_ADDRESS(ptr_imKF+1) +
                        w_ref_bl * VALUE_FROM_ADDRESS(ptr_imKF+step) +
                        w_ref_br * VALUE_FROM_ADDRESS(ptr_imKF+step+1);
                    float res = (A*VALUE_FROM_ADDRESS(ptrimKF+cur_idx)+B - last_intensity);
                    //LOG(INFO) << "J = " << J;

                    J(0, 4) = -VALUE_FROM_ADDRESS(ptrimKF+cur_idx);
                    J(0, 5) = -1.;
                    if (fabs(res) > g_robust_delta)
                        w *= g_robust_delta/fabs(res);
                    imHessian += J.transpose() * J * w;
                    JTe += res * J.transpose() * w;
                    chi2KF += res * res;
                }
            }

        if (npixelKF == 0){
            LOG(FATAL) << "no overlap";
        }

        float avgChi2KF = npixelKF == 0?1:chi2KF/npixelKF;
        if (use_gn){
            imHessian(0, 0) += g_weight_opt_cf_p;
            imHessian(1, 1) += g_weight_opt_cf_p;
            imHessian(2, 2) += g_weight_opt_cf_p;
            imHessian(3, 3) += g_weight_opt_cf_yaw;
            imHessian(4, 4) += g_weight_opt_cf_ab;
            imHessian(5, 5) += g_weight_opt_cf_ab;
            Vector6 delta = imHessian.ldlt().solve(JTe);
            LOG(INFO) << "imHessan: " << imHessian;
            LOG(INFO) << "JTe: " << JTe.transpose();
            pKFCur->mPos += delta.template segment<3>(0);
            psi += delta(3);
            A += delta(4);
            B += delta(5);
            typename NormalVector::MPD rotMatrix;
            rotMatrix.setMatrix(RotZT(psi)*Rn.transpose());
            pKFCur->mRot = rotMatrix;
            LOG(INFO) << "avgChi2KFOp: " << avgChi2KF << " normd: " << delta.norm() << " psi: " << psi << " mPos: " << pKFCur->mPos.transpose() << " delta: " << delta.transpose();

            if (g_draw_homography){
                cv::Mat homoImg = DrawRect(ICur, ILoop, Hkf);
                Scalar gradient = GradientMagRMS(IxCur, IyCur);
                char buf[256];

                // sprintf(buf, "gra: %.2f", gradient);
                // cv::putText(homoImg, buf, cv::Point(15, 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

                sprintf(buf, "id: %d", pKFLoop->mnFrameId);
                cv::putText(homoImg, buf, cv::Point(5, 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

                sprintf(buf, "c2: %.2f", avgChi2KF);
                cv::putText(homoImg, buf, cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

                int scale2 = scale*100;
                sprintf(buf, "/tmp/homorect_%06d_%06d_%02d_%02d_optimizecf.png", pKFCur->mnFrameId, pKFLoop->mnFrameId, scale2, iter);
                cv::imwrite(buf, homoImg);
                // homoImg = DrawRect(ICur, ILoop, Hkf2);
                // sprintf(buf, "/tmp/homorect_%06d_%02d_optimizecf2.png", pKFCur->mnFrameId, iter);
                // cv::imwrite(buf, homoImg);
                LOG(INFO) << std::string(buf);
            }
            if(delta.norm() < 1e-3 || iter++ >= maxIter || avgChi2KF < 0.01){
                return avgChi2KF;
            }
        }else{
            int qiter = 0;
            Scalar psi0 = psi;
            Vector3 pos0 = pKFCur->mPos;
            float A0 = A;
            float B0 = B;
            float avgChi2KF2 = 0.;
            do{
                imHessian.diagonal().array() += lambda;
                float avgChi2KF = chi2KF/npixelKF;
                Vector6 delta = imHessian.ldlt().solve(JTe);
                LOG(INFO) << "imHessan: " << imHessian;
                LOG(INFO) << "JTe: " << JTe.transpose();
                pKFCur->mPos += delta.template segment<3>(0);
                psi += delta(3);
                A += delta(4);
                B += delta(5);
                typename NormalVector::MPD rotMatrix;
                rotMatrix.setMatrix(RotZT(psi)*Rn.transpose());
                pKFCur->mRot = rotMatrix;
                LOG(INFO) << "avgChi2KFOp: " << avgChi2KF << " normd: " << delta.norm() << " psi: " << psi << " mPos: " << pKFCur->mPos.transpose() << " delta: " << delta.transpose();

                Rpsi = RotZ(psi);
                Rcw = Rn*Rpsi;
                Rwc = Rcw.transpose();
                RKFc = RKFw*Rwc;
                posKFc = RKFw*(pKFCur->mPos - pKFLoop->mPos);
                Matrix3 Hkf3 = K*(RKFc + pKFCur->mdInverseDistance*posKFc*n.transpose())*Kinv;

                RcKF = RKFc.transpose();
                wrcKF = pKFLoop->mPos - pKFCur->mPos;
                poscKF = Rcw*wrcKF;
                Matrix3 Hkf4 = K*(RcKF + pKFLoop->mdInverseDistance * poscKF * n2.transpose())*Kinv;

                // recalculate intensity error
                int npixelKF2 = 0;
                float chi2KF2 = 0.f;
                for(int y = border; y<ICur.rows-border; y++)
                    for(int x = border; x<ICur.cols-border; x++) {
                        const Vector3 p(x, y, 1);
                        Vector3 pk_1homo = Hkf3*p;
                        Vector2 pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
                        if (InsideImage(pkf(0), pkf(1), ICur.rows, ICur.cols)){
                            npixelKF2++;
                            int last_u_i = static_cast<int>(floor(pkf(0)));
                            int last_v_i = static_cast<int>(floor(pkf(1)));
                            const float subpix_u_ref = pkf(0) - last_u_i;
                            const float subpix_v_ref = pkf(1) - last_v_i;
                            const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                            const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                            const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                            const float w_ref_br = subpix_u_ref * subpix_v_ref;
                            int cur_u = x;
                            int cur_v = y;
                            int last_idx = last_u_i + last_v_i*step;
                            int cur_idx = cur_u + cur_v*step;

                            const float *ptr_imKF = ptrimKF + last_idx;
                            float last_intensity = w_ref_tl * VALUE_FROM_ADDRESS(ptr_imKF) +
                                w_ref_tr * VALUE_FROM_ADDRESS(ptr_imKF+1) +
                                w_ref_bl * VALUE_FROM_ADDRESS(ptr_imKF+step) +
                                w_ref_br * VALUE_FROM_ADDRESS(ptr_imKF+step+1);
                            float res = (VALUE_FROM_ADDRESS(ptrimk+cur_idx) - A*last_intensity - B);
                            chi2KF2 += res * res;
                        }

                        pk_1homo = Hkf4*p;
                        pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
                        if (InsideImage(pkf(0), pkf(1), ICur.rows, ICur.cols)){
                            npixelKF2++;
                            int last_u_i = static_cast<int>(floor(pkf(0)));
                            int last_v_i = static_cast<int>(floor(pkf(1)));
                            const float subpix_u_ref = pkf(0) - last_u_i;
                            const float subpix_v_ref = pkf(1) - last_v_i;
                            const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                            const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                            const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                            const float w_ref_br = subpix_u_ref * subpix_v_ref;
                            int cur_u = x;
                            int cur_v = y;
                            int last_idx = last_u_i + last_v_i*step;
                            int cur_idx = cur_u + cur_v*step;

                            const float *ptr_imKF = ptrimk + last_idx;
                            float last_intensity = 
                                w_ref_tl * VALUE_FROM_ADDRESS(ptr_imKF) +
                                w_ref_tr * VALUE_FROM_ADDRESS(ptr_imKF+1) +
                                w_ref_bl * VALUE_FROM_ADDRESS(ptr_imKF+step) +
                                w_ref_br * VALUE_FROM_ADDRESS(ptr_imKF+step+1);
                            float res = (A*VALUE_FROM_ADDRESS(ptrimKF+cur_idx)+B - last_intensity);
                            chi2KF2 += res * res;
                        }
                    }
                float avgChi2KF2 = chi2KF2/npixelKF2;
                if (avgChi2KF2 < avgChi2KF){
                    LOG(INFO) << "lambda: " << lambda << " avgChi2KF2: " << avgChi2KF2;
                    lambda /= 2;
                    mu = 2;
                    break;
                }else{
                    imHessian.diagonal().array() -= lambda;
                    lambda *= mu;
                    mu *= 2;
                    psi = psi0;
                    pKFCur->mPos = pos0;
                    A = A0;
                    B = B0;
                }
            }while(qiter++ < innerMaxIter);

            if (g_draw_homography){
                cv::Mat homoImg = DrawRect(ICur, ILoop, Hkf);
                Scalar gradient = GradientMagRMS(IxCur, IyCur);
                char buf[256];

                // sprintf(buf, "gra: %.2f", gradient);
                // cv::putText(homoImg, buf, cv::Point(15, 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

                sprintf(buf, "kfid: %d", pKFLoop->mnFrameId);
                cv::putText(homoImg, buf, cv::Point(15, 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

                sprintf(buf, "chi2: %.2f", avgChi2KF);
                cv::putText(homoImg, buf, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255));

                sprintf(buf, "/tmp/homorect_%06d_%06d_%02d_optimizecf.png", pKFCur->mnFrameId, pKFLoop->mnFrameId, iter);
                cv::imwrite(buf, homoImg);
                // homoImg = DrawRect(ICur, ILoop, Hkf2);
                // sprintf(buf, "/tmp/homorect_%06d_%02d_optimizecf2.png", pKFCur->mnFrameId, iter);
                // cv::imwrite(buf, homoImg);
                LOG(INFO) << std::string(buf);

            }

            if(iter++ >= maxIter || avgChi2KF < 0.01 || qiter >= innerMaxIter){
                return avgChi2KF;
            }
        }
    }
}

template <typename Scalar, int _D>
float State<Scalar, _D>::AverageIntensityChi2(int kfid){
    Frame* pKF = mvpKfs[kfid];
    Vector3 ez(0, 0, 1.);
    float chi2 = 0.;
    int border = 3;
    const int step = mKF.mImg.cols;
    // initialization for psi
    Matrix3 Rwck = typename NormalVector::MPD(mRot).matrix();
    Matrix3 Rwck_1 = typename NormalVector::MPD(pKF->mRot).matrix();
    Matrix3 Rcwk = Rwck.transpose();
    Matrix3 Rcwk_1 = Rwck_1.transpose();

    const float *ptrimk = imk.ptr<float>(0, 0);
    const float *ptrimKF = pKF->mImg.ptr<float>(0, 0);

    const Vector3 &n = mUnitDirection.getVec();
    const double alpha = mfAlpha;

    Matrix3 Rk_1k = Rcwk_1*Rwck;
    Vector3 t12 = Rcwk_1*(mPos - pKF->mPos);
    Matrix3 Hkf = mK*(Rk_1k + alpha*t12*n.transpose())*mKinv;
    int npixel = 0;

    for(int y = border; y<pKF->mImg.rows-border; y++)
        for(int x = border; x<pKF->mImg.cols-border; x++) {
            const Vector3 p(x, y, 1);
            const Vector3 pk_1homo = Hkf*p;
            const Vector2 pkf = pk_1homo.template segment<2>(0)/pk_1homo(2);
            if (InsideImage(pkf(0), pkf(1), imk_1.rows, imk_1.cols)){
                npixel++;
                // compute image derivative w.r.t x and y direction
                int last_u_i = static_cast<int>(floor(pkf(0)));
                int last_v_i = static_cast<int>(floor(pkf(1)));
                const float subpix_u_ref = pkf(0) - last_u_i;
                const float subpix_v_ref = pkf(1) - last_v_i;
                const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                const float w_ref_br = subpix_u_ref * subpix_v_ref;
                int cur_u = x;
                int cur_v = y;
                int last_idx = last_u_i + last_v_i*step;
                int cur_idx = cur_u + cur_v*step;

                //intensity residue
                const float *ptr_imKF = ptrimKF + last_idx;
                float last_intensity = w_ref_tl * VALUE_FROM_ADDRESS(ptr_imKF) +
                    w_ref_tr * VALUE_FROM_ADDRESS(ptr_imKF+1) +
                    w_ref_bl * VALUE_FROM_ADDRESS(ptr_imKF+step) +
                    w_ref_br * VALUE_FROM_ADDRESS(ptr_imKF+step+1);
                float res = (VALUE_FROM_ADDRESS(ptrimk+cur_idx) - last_intensity);
                chi2 += res * res;
            }
        }
    return npixel==0?0.:chi2/npixel;
}

template <typename Scalar, int _D>
Scalar State<Scalar, _D>::GradientMagRMS(const cv::Mat &Ix, const cv::Mat &Iy){
    return sqrt((cv::sum(Ix.mul(Ix))[0] + cv::sum(Iy.mul(Iy))[0])/(Ix.rows*Ix.cols));
}

template <typename Scalar, int _D>
int State<Scalar, _D>::SearchOverlapKF(){
    // find the highest iou kf and greater than a threshold
    Vector3 n = mUnitDirection.getVec();
    int maxIdx = -1;
    float maxIou = -1.f;
    Matrix3 Rwc = typename NormalVector::MPD(mRot).matrix();
    std::vector<std::pair<int, float> > kfCandidatesIdx;
    for (int i = 0, iend = mvpKfs.size()-1; i < iend; i++){
        Frame* pkf = mvpKfs[i];
        if (pkf->mbBad){
            continue;
        }
        const typename NormalVector::QPD &rotKF = pkf->mRot;
        const Vector3 &posKF = pkf->mPos;
        Matrix3 RKFw = typename NormalVector::MPD(rotKF).matrix().transpose();
        Vector3 PosKF = -RKFw*posKF;
        Matrix3 RKFc = RKFw*Rwc;
        Vector3 posKFc = RKFw*mPos + PosKF;
        Matrix3 Hkf = mK*(RKFc+mfAlpha*posKFc*n.transpose())*mKinv;
        std::vector<cv::Point2f> srcPts, dstPts;
        HomograpyMapImagePoints(Hkf, srcPts, dstPts, imk_1.rows, imk_1.cols);
        float iou = PolygonIntersection::IOU(srcPts, dstPts);
        LOG(INFO) << "kf id: " << pkf->mnFrameId << " search iou: " << iou;
        if (iou > g_kf_iou_th){
            kfCandidatesIdx.emplace_back(i, iou);
            // maxIou = iou;
            // maxIdx = i;
        }
    }
    std::sort(kfCandidatesIdx.begin(), kfCandidatesIdx.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b){
            return a.second > b.second;
            });

    Frame tmpF;
    tmpF.mnFrameId = mnFrameId;
    imk.copyTo(tmpF.mImg);
    Ixk.copyTo(tmpF.mIx);
    Iyk.copyTo(tmpF.mIy);
    tmpF.mdInverseDistance = mfAlpha;
    tmpF.mNormal = mUnitDirection.getVec();
    Frame* pKFNew = &tmpF;
    for (size_t i = 0, iend = kfCandidatesIdx.size(); i<iend; i++){
        tmpF.mRot = mRot;
        tmpF.mPos = mPos;
        int kfidx = kfCandidatesIdx[i].first;
        float iou = kfCandidatesIdx[i].second;

        Frame* pKFLoop = mvpKfs[kfidx];
        float avgChi2Loop = AverageIntensityChi2(kfidx);
        pKFLoop->mnTrackedCandidate++;
        LOG(INFO) << "kfidx: " << kfidx << " iou: " << iou;
        if (avgChi2Loop > 0.05){
            Vector3 pos0 = pKFNew->mPos;
            avgChi2Loop = OptimizeCF(pKFLoop, pKFNew, g_opt_cf_kf_scale);
            float normDistance = (pos0 - pKFNew->mPos).norm();
            if (avgChi2Loop < 0.03 && normDistance < g_opt_cf_pos_th){
                mRot = tmpF.mRot;
                mPos = tmpF.mPos;
                LOG(INFO) << "accept normdistance: " << normDistance;
                pKFLoop->mnTrackedSuccess++;
                return kfidx;
            }
        }else{
            pKFLoop->mnTrackedSuccess++;
            return kfidx;
        }
        float successRate = static_cast<float>(pKFLoop->mnTrackedSuccess)/static_cast<float>(pKFLoop->mnTrackedCandidate);
        
        LOG(INFO) << "success rate: " <<  successRate << " iou: " << kfCandidatesIdx[i].second;
        if (pKFLoop->mnTrackedCandidate > 5 && successRate < 0.2){
            pKFLoop->mbBad = true;
        }
    }
    return -1;

}

template <typename Scalar, int _D>
void State<Scalar, _D>::HomograpyMapImagePoints(const Matrix3 &H, std::vector<cv::Point2f> &srcPts, std::vector<cv::Point2f> &dstPts, int rows, int cols){
    srcPts.clear();
    dstPts.clear();
    Vector3 tl(0, 0, 1.), tr(cols, 0, 1.), bl(0, rows, 1.), br(cols, rows, 1.);
    auto homographyTransform = [](const Matrix3 &H, const Vector3 &p, std::vector<cv::Point2f> &src){
        Vector3 tmp = H*p;
        tmp /= tmp(2);
        // LOG(INFO) << "pt: " << p.transpose() << " homo pt: " << tmp.transpose();
        src.emplace_back(p[0]/p[2], p[1]/p[2]);
        return cv::Point2f(tmp[0], tmp[1]);
    };
    dstPts.emplace_back(homographyTransform(H, tl, srcPts));
    dstPts.emplace_back(homographyTransform(H, bl, srcPts));
    dstPts.emplace_back(homographyTransform(H, br, srcPts));
    dstPts.emplace_back(homographyTransform(H, tr, srcPts));
}

template <typename Scalar, int _D>
cv::Mat State<Scalar, _D>::DrawRect(const cv::Mat &im1, const cv::Mat &im2, const Matrix3 &H){
    std::vector<cv::Point2f> srcPts, dstPts;
    HomograpyMapImagePoints(H, srcPts, dstPts, im1.rows, im1.cols);
    return DrawRect2(im1, im2, srcPts, dstPts);
}
