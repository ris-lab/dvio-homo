//
// Created by Zhong,Shangkun on 2018/9/20.
//

#include "setting_config.h"
#include <glog/logging.h>
#include <iostream>
int g_origin_frame_cols = 1280;
int g_origin_frame_rows = 720;
int g_frame_cols = 640;
int g_frame_rows = 480;
float g_fx = 500.f;
float g_fy = 500.f;
float g_cx = 300.f;
float g_cy = 200.f;
float g_k1 = 0.f;
float g_k2 = 0.f;
float g_p1 = 0.f;
float g_p2 = 0.f;
float g_new_fx = 450.f;
float g_new_fy = 450.f;
float g_new_cx = 320.f;
float g_new_cy = 220.f;
float g_scale = 1.f;
float g_im_weight = 1.f;
float g_sigma_az = 1.f;
float g_sigma_ay = 1.f;
float g_sigma_ax = 1.f;

float g_sigma_ba = 1.f;
float g_sigma_wx = 1.f;
float g_sigma_wy = 1.f;
float g_sigma_wz = 1.f;
float g_sigma_bw = 1.f;

int g_max_iteration = 3;
int g_use_lk = 0;

int g_use_Tci = 1;
cv::Mat g_Tci = cv::Mat::eye(4,4,CV_32FC1);
cv::Mat g_Tic = cv::Mat::eye(4,4,CV_32FC1);
int g_level = 0;
double g_time_shift = 0;

float g_sigma_pos;
float g_sigma_vel;
float g_sigma_alpha;
float g_sigma_n;
float g_sigma_att;
float g_sigma_A;
float g_sigma_B;

float g_sigma_alpha0;
float g_sigma_pos0;
float g_sigma_att0;
float g_sigma_vel0;
float g_sigma_n0;
float g_sigma_ba0;
float g_sigma_bw0;
float g_sigma_A0;
float g_sigma_B0;
float g_sigma_yaw;
float g_robust_delta = 0.2;
int g_show_log;
int g_use_huber;
int g_draw_homography = 0;
float g_kf_coverage = 0.3;
int g_init_nkf = 10;
float g_gaussian_sigma = 3;

float g_im_weight_kf = 1.;
int g_use_kf = 0;
int g_nframes = -1;
float g_kf_iou_th = 0.5;
int g_search_kf = 1;
int g_change_weight = 1;
float g_im_weight2 = 10.f;
float g_im_weight_kf2 = 30.f;
int g_marginalize_AB = 1;
int g_marginalize_AB_c = 1;
int g_optimize_kf_max_iter = 10;
int g_use_median_blur = 0;
int g_kf_match_th = 10;
float g_opt_cf_kf_scale = 1.;
float g_weight_opt_cf_p = 1.;
float g_weight_opt_cf_ab = 1.;
float g_weight_opt_cf_yaw = 1.;
float g_opt_cf_pos_th = 0.1;
float g_init_alpha0 = 10.f;
int g_fisheye = 0;
float g_avg_kf_grad_th = 1.;

void ReadGlobalParaFromYaml(const std::string &paraDir){
    cv::FileStorage fsSettings(paraDir.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << paraDir << std::endl;
        exit(-1);
    }
    fsSettings["Init.Alpha0"] >> g_init_alpha0;
    fsSettings["Config.FrameRows"] >> g_frame_rows;
    fsSettings["Config.FrameCols"] >> g_frame_cols;
    fsSettings["Config.OriginFrameRows"] >> g_origin_frame_rows;
    fsSettings["Config.OriginFrameCols"] >> g_origin_frame_cols;
    fsSettings["Config.DownSample"] >> g_level;
    fsSettings["Config.Scale"] >> g_scale;
    fsSettings["Config.ImageWeight"] >> g_im_weight;
    fsSettings["Config.ImageWeightKF"] >> g_im_weight_kf;
    fsSettings["Config.MaxIteration"] >> g_max_iteration;
    fsSettings["Config.UseLK"] >> g_use_lk;
    fsSettings["Config.SigmaAx"] >> g_sigma_ax;
    fsSettings["Config.SigmaAy"] >> g_sigma_ay;
    fsSettings["Config.SigmaAz"] >> g_sigma_az;
    fsSettings["Config.SigmaBa"] >> g_sigma_ba;
    fsSettings["Config.SigmaWx"] >> g_sigma_wx;
    fsSettings["Config.SigmaWy"] >> g_sigma_wy;
    fsSettings["Config.SigmaWz"] >> g_sigma_wz;
    fsSettings["Config.SigmaBw"] >> g_sigma_bw;
    fsSettings["Config.SigmaAlpha"] >> g_sigma_alpha;
    fsSettings["Config.SigmaVel"] >> g_sigma_vel;
    fsSettings["Config.SigmaPos"] >> g_sigma_pos;
    fsSettings["Config.SigmaN"] >> g_sigma_n;
    fsSettings["Config.SigmaA"] >> g_sigma_A;
    fsSettings["Config.SigmaB"] >> g_sigma_B;
    fsSettings["Config.SigmaAtt"] >> g_sigma_att;
    fsSettings["Config.SigmaYaw"] >> g_sigma_yaw;
    fsSettings["Config.UseHuber"] >> g_use_huber;
    fsSettings["Config.UseKF"] >> g_use_kf;

    fsSettings["Config.SigmaAlpha0"] >> g_sigma_alpha0;
    fsSettings["Config.SigmaPos0"] >> g_sigma_pos0;
    fsSettings["Config.SigmaVel0"] >> g_sigma_vel0;
    fsSettings["Config.SigmaN0"] >> g_sigma_n0;
    fsSettings["Config.SigmaAtt0"] >> g_sigma_att0;
    fsSettings["Config.SigmaBa0"] >> g_sigma_ba0;
    fsSettings["Config.SigmaBw0"] >> g_sigma_bw0;
    fsSettings["Config.SigmaA0"] >> g_sigma_A0;
    fsSettings["Config.SigmaB0"] >> g_sigma_B0;
    fsSettings["Config.RobustDelta"] >> g_robust_delta;
    fsSettings["Config.KFCoverage"] >> g_kf_coverage;
    fsSettings["Config.NFrames"] >> g_nframes;

    fsSettings["Camera.fx"] >> g_fx;
    fsSettings["Camera.fy"] >> g_fy;
    fsSettings["Camera.cx"] >> g_cx;
    fsSettings["Camera.cy"] >> g_cy;

    fsSettings["Camera.k1"] >> g_k1;
    fsSettings["Camera.k2"] >> g_k2;
    fsSettings["Camera.p1"] >> g_p1;
    fsSettings["Camera.p2"] >> g_p2;
    fsSettings["Camera.FishEye"] >> g_fisheye;

    fsSettings["Camera.newfx"] >> g_new_fx;
    fsSettings["Camera.newfy"] >> g_new_fy;
    fsSettings["Camera.newcx"] >> g_new_cx;
    fsSettings["Camera.newcy"] >> g_new_cy;

    fsSettings["CameraIMU.bTci"] >> g_use_Tci;
    fsSettings["Config.ShowLog"] >> g_show_log;
    fsSettings["Config.DrawHomography"] >> g_draw_homography;
    fsSettings["Config.InitNKF"] >> g_init_nkf;
    fsSettings["Config.GaussianSigma"] >> g_gaussian_sigma;
    fsSettings["Config.KFIouTh"] >> g_kf_iou_th;
    fsSettings["Config.SearchKF"] >> g_search_kf;
    fsSettings["Config.ChangeWeight"] >> g_change_weight;
    fsSettings["Config.ImageWeight2"] >> g_im_weight2;
    fsSettings["Config.ImageWeightKF2"] >> g_im_weight_kf2;
    fsSettings["Config.MarginalizeAB"] >> g_marginalize_AB;
    fsSettings["Config.MarginalizeABC"] >> g_marginalize_AB_c;
    fsSettings["Config.OptimizeKFMaxIter"] >> g_optimize_kf_max_iter;
    fsSettings["Config.UseMedianBlur"] >> g_use_median_blur;
    fsSettings["Config.KFMatchTh"] >> g_kf_match_th;
    fsSettings["Config.OptCFKFScale"] >> g_opt_cf_kf_scale;
    fsSettings["Config.WeightOptCFP"] >> g_weight_opt_cf_p;
    fsSettings["Config.WeightOptCFAB"] >> g_weight_opt_cf_ab;
    fsSettings["Config.WeightOptCFYaw"] >> g_weight_opt_cf_yaw;
    fsSettings["Config.OptCFPosTh"] >> g_opt_cf_pos_th;
    fsSettings["Config.AvgKFGradTh"] >> g_avg_kf_grad_th;

	if(g_use_Tci > 0){
		cv::Mat Tci;
		fsSettings["CameraIMU.T"] >> Tci;
		g_Tci = cv::Mat::eye(4, 4, CV_32FC1);
		Tci.copyTo(g_Tci.rowRange(0, 3));
		g_Tic.rowRange(0, 3).colRange(0, 3) = g_Tci.rowRange(0, 3).colRange(0, 3).t();
		g_Tic.rowRange(0, 3).col(3) = -g_Tic.rowRange(0, 3).colRange(0, 3)*g_Tci.rowRange(0, 3).col(3);
	}
	else{
		cv::Mat Tic;
		fsSettings["CameraIMU.T"] >> Tic;
		g_Tic = cv::Mat::eye(4, 4, CV_32FC1);
		Tic.copyTo(g_Tic.rowRange(0, 3));
		g_Tci.rowRange(0, 3).colRange(0, 3) = g_Tic.rowRange(0, 3).colRange(0, 3).t();
		g_Tci.rowRange(0, 3).col(3) = -g_Tci.rowRange(0, 3).colRange(0, 3)*g_Tic.rowRange(0, 3).col(3);
	}
    fsSettings["CameraIMU.TimeShift"] >> g_time_shift;
    LOG(INFO) << "- fx: " << g_fx;
    LOG(INFO) << "- fy: " << g_fy;
    LOG(INFO) << "- cx: " << g_cx;
    LOG(INFO) << "- cy: " << g_cy;
    LOG(INFO) << "- time shift: " << g_time_shift;
    LOG(INFO) << "- k1: " << g_k1;
    LOG(INFO) << "- k2: " << g_k2;
    LOG(INFO) << "- p1: " << g_p1;
    LOG(INFO) << "- p2: " << g_p2;
    LOG(INFO) << "- new fx: " << g_new_fx;
    LOG(INFO) << "- new fy: " << g_new_fy;
    LOG(INFO) << "- new cx: " << g_new_cx;
    LOG(INFO) << "- new cy: " << g_new_cy;
    LOG(INFO) << "- fish eye: " << g_fisheye;
    LOG(INFO) << "- init_alpha0: " << g_init_alpha0;
    LOG(INFO) << "- sigma_ax: " << g_sigma_ax;
    LOG(INFO) << "- sigma_ay: " << g_sigma_ay;
    LOG(INFO) << "- sigma_az: " << g_sigma_az;
    LOG(INFO) << "- sigma_ba: " << g_sigma_ba;
    LOG(INFO) << "- sigma_wx: " << g_sigma_wx;
    LOG(INFO) << "- sigma_wy: " << g_sigma_wy;
    LOG(INFO) << "- sigma_wz: " << g_sigma_wz;
    LOG(INFO) << "- sigma_bw: " << g_sigma_bw;
    LOG(INFO) << "- sigma_alpha: " << g_sigma_alpha;
    LOG(INFO) << "- sigma_n: " << g_sigma_n;
    LOG(INFO) << "- sigma_att: " << g_sigma_att;
    LOG(INFO) << "- sigma_vel: " << g_sigma_vel;
    LOG(INFO) << "- sigma_pos: " << g_sigma_pos;
    LOG(INFO) << "- sigma_a: " << g_sigma_A;
    LOG(INFO) << "- sigma_b: " << g_sigma_B;
    LOG(INFO) << "- sigma_yaw: " << g_sigma_yaw;

    LOG(INFO) << "- sigma_alpha0: " << g_sigma_alpha0;
    LOG(INFO) << "- sigma_n0: " << g_sigma_n0;
    LOG(INFO) << "- sigma_att0: " << g_sigma_att0;
    LOG(INFO) << "- sigma_pos0: " << g_sigma_pos0;
    LOG(INFO) << "- sigma_vel0: " << g_sigma_vel0;
    LOG(INFO) << "- sigma_ba0: " << g_sigma_ba0;
    LOG(INFO) << "- sigma_bw0: " << g_sigma_bw0;
    LOG(INFO) << "- sigma_a0: " << g_sigma_A0;
    LOG(INFO) << "- sigma_b0: " << g_sigma_B0;
    LOG(INFO) << "- draw_homography: " << g_draw_homography;
    LOG(INFO) << "- kf_coverage: " << g_kf_coverage;
    LOG(INFO) << "- init_nkf: " << g_init_nkf;
    LOG(INFO) << "- gassian_sigma: " << g_gaussian_sigma;
    LOG(INFO) << "- im_weight_kf: " << g_im_weight_kf;
    LOG(INFO) << "- im_weight: " << g_im_weight;
    LOG(INFO) << "- use_kf: " << g_use_kf;
    LOG(INFO) << "- nframes: " << g_nframes;
    LOG(INFO) << "- kf_iou_th: " << g_kf_iou_th;
    LOG(INFO) << "- search_kf: " << g_search_kf;
    LOG(INFO) << "- change_weight: " << g_change_weight;
    LOG(INFO) << "- im_weight: " << g_im_weight2;
    LOG(INFO) << "- im_weight_kf2: " << g_im_weight_kf2;
    LOG(INFO) << "- marginalize_AB: " << g_marginalize_AB;
    LOG(INFO) << "- marginalize_AB_c: " << g_marginalize_AB_c;
    LOG(INFO) << "- optimize_kf_max_iter: " << g_optimize_kf_max_iter;
    LOG(INFO) << "- use_median_blur: " << g_use_median_blur;
    LOG(INFO) << "- kf_match_th: " << g_kf_match_th;
    LOG(INFO) << "- opt_cf_kf_scale: " << g_opt_cf_kf_scale;
    LOG(INFO) << "- weight_opt_cf_p: " << g_weight_opt_cf_p;
    LOG(INFO) << "- weight_opt_cf_ab: " << g_weight_opt_cf_ab;
    LOG(INFO) << "- weight_opt_cf_yaw: " << g_weight_opt_cf_yaw;
    LOG(INFO) << "- opt_cf_pos_th: " << g_opt_cf_pos_th;
    LOG(INFO) << "- g_avg_kf_grad_th: " << g_avg_kf_grad_th;
}
