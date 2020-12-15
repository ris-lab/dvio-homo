//
// Created by Zhong,Shangkun on 2018/9/20.
//

#ifndef ORB_SLAM2_VIO_MACRO_H
#define ORB_SLAM2_VIO_MACRO_H

#include <sys/time.h>

#define debug_code(x) x
#define print_info(v) std::cout<<v<<std::endl;
#define print_info2(v1, v2) std::cout<<v1<<"\t"<<v2<<std::endl;
#define print_info3(v1, v2, v3) std::cout<<v1<<"\t"<<v2<<"\t"<<v3<<std::endl;
#define print_info4(v1, v2, v3, v4) std::cout<<v1<<"\t"<<v2<<"\t"<<v3<<"\t"<<v4<<std::endl;
#define print_info5(v1, v2, v3, v4, v5) std::cout<<v1<<"\t"<<v2<<"\t"<<v3<<"\t"<<v4<<"\t"<<v5<<std::endl;
#define print_error(v) std::cerr<<"[ERROR] "<<v<<std::endl;
#define print_info_stream(x) std::cerr<<"[INFO] "<<x<<std::endl;
#define print_debug_stream(x) print_info_stream(x)
#define print_warn_stream(x) std::cerr<<"[WARN] "<<x<<std::endl;

inline double calc_timer(const timeval& timer_s) {
    timeval timer_e;
    gettimeofday(&timer_e, NULL);
    return (timer_e.tv_sec - timer_s.tv_sec) * 1000 + (timer_e.tv_usec - timer_s.tv_usec) / 1000.0;
}
#define START_TIMER_MS(timer_s) timeval timer_s; gettimeofday(&timer_s, NULL);
#define STOP_TIMER_MS(timer_s) calc_timer(timer_s);
#define STOP_TIMER_MS_STR(v, timer_s) std::cout<<"[TIME ANALYSIS (ms)] "<< v << calc_timer(timer_s) <<std::endl;


#define START_CV_TIME(time) double time = cv::getTickCount()
#define END_CV_TIME_MS(time) time = (cv::getTickCount() - time)/cv::getTickFrequency()*1e3
#define END_CV_TIME_S(time) time = (cv::getTickCount() - time)/cv::getTickFrequency()
#define LOG_END_CV_TIME_MS(time) LOG(INFO) << "[Time Cost] " << #time << " = "<< (cv::getTickCount() - time)/cv::getTickFrequency()*1e3 << " ms"
#define LOG_END_CV_TIME_S(time) LOG(INFO) << "[Time Cost] " << #time << " = "<< (cv::getTickCount() - time)/cv::getTickFrequency() << " s"
#define LOG_INFO_1(v1) LOG(INFO) << #v1 << " = " << v1
#define LOG_INFO_2(v1, v2) LOG(INFO) << #v1 << " = " << v1 << " " << #v2 << " = " << v2
#define LOG_INFO_3(v1, v2, v3) LOG(INFO) << #v1 << " = " << v1 << " " << #v2 << " = " << v2 << " " << #v3 << " = " << v3
#define LOG_INFO_4(v1, v2, v3, v4) LOG(INFO) << #v1 << " = " << v1 << " " << #v2 << " = " << v2 << " " << #v3 << " = " << v3 << " " << #v4 << " = " << v4
#define LOG_INFO_5(v1, v2, v3, v4, v5) LOG(INFO) << #v1 << " = " << v1 << " " << #v2 << " = " << v2 << " " << #v3 << " = " << v3 << " " << #v4 << " = " << v4 << " " << #v5 << " = " << v5

#ifdef __ANDROID__
// log info for android
#include <android/log.h>
#define ANDROID_LOG_TAG "[VIO LOG]"
#define LOGV(...)__android_log_print(ANDROID_LOG_VERBOSE, ANDROID_LOG_TAG, __VA_ARGS__)// VERBOSE
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, ANDROID_LOG_TAG, __VA_ARGS__)// DEBUG
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, ANDROID_LOG_TAG,__VA_ARGS__)// INFO
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, ANDROID_LOG_TAG, __VA_ARGS__)//WARN
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, ANDROID_LOG_TAG,__VA_ARGS__)// ERROR
#else
#include <stdio.h>
#define LOGE(format__,...) printf(format__, __VA_ARGS__)
#endif
#endif //ORB_SLAM2_VIO_MACRO_H
