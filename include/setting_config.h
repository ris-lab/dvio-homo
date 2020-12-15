//
// Created by zsk on 2018/11/31.
//

#ifndef SETTING_CONFIG_H
#define SETTING_CONFIG_H

#include <string>
#include <opencv2/core/core.hpp>
extern int g_origin_frame_cols;
extern int g_origin_frame_rows;
extern int g_frame_cols;
extern int g_frame_rows;
extern float g_fx;
extern float g_fy;
extern float g_cx;
extern float g_cy;
extern float g_k1;
extern float g_k2;
extern float g_p1;
extern float g_p2;
extern int g_use_Tci;
extern cv::Mat g_Tci;
extern cv::Mat g_Tic;
extern int g_level;
extern float g_scale;
extern double g_time_shift;
extern float g_im_weight;
extern int g_max_iteration;
extern int g_use_lk;
extern float g_init_alpha0;
extern int g_fisheye;

extern float g_sigma_pos;
extern float g_sigma_vel;
extern float g_sigma_att;
extern float g_sigma_ax;
extern float g_sigma_ay;
extern float g_sigma_az;
extern float g_sigma_ba;
extern float g_sigma_wx;
extern float g_sigma_wy;
extern float g_sigma_wz;
extern float g_sigma_bw;
extern float g_sigma_alpha;
extern float g_sigma_n;
extern float g_sigma_A;
extern float g_sigma_B;
extern float g_sigma_yaw;

extern float g_sigma_alpha0;
extern float g_sigma_pos0;
extern float g_sigma_vel0;
extern float g_sigma_att0;
extern float g_sigma_n0;
extern float g_sigma_ba0;
extern float g_sigma_bw0;
extern float g_sigma_A0;
extern float g_sigma_B0;

extern float g_robust_delta;
extern int g_show_log;
extern int g_use_huber;
extern int g_use_kf;
extern float g_new_fx;
extern float g_new_fy;
extern float g_new_cx;
extern float g_new_cy;
extern int g_draw_homography;
extern float g_kf_coverage;
extern int g_init_nkf;
extern float g_gaussian_sigma;
extern float g_im_weight_kf;
extern int g_nframes;
extern float g_kf_iou_th;
extern int g_use_average_imu;
extern int g_search_kf;
extern int g_change_weight;
extern float g_im_weight2;
extern float g_im_weight_kf2;
extern int g_marginalize_AB;
extern int g_marginalize_AB_c;
extern int g_optimize_kf_max_iter;
extern int g_use_median_blur;
extern int g_kf_match_th;
extern float g_opt_cf_kf_scale;
extern float g_weight_opt_cf_p;
extern float g_weight_opt_cf_ab;
extern float g_weight_opt_cf_yaw;
extern float g_opt_cf_pos_th;
extern float g_avg_kf_grad_th;
void ReadGlobalParaFromYaml(const std::string &paraDir);

#endif //ORB_SLAM2_SETTING_CONFIG_H
