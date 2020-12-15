#include "VideoReader.h"
#include "State.h"
#include <Eigen/Geometry>
#include <sstream>
#include "vio_macro.h"
#include <opencv2/core/eigen.hpp>
#include <gflags/gflags.h>
#include <boost/format.hpp>
using namespace std;
DEFINE_string(yaml_path, "bebop_bottom.yaml", "yaml config");
DEFINE_string(data_path, "data/", "video and imu path for test");
DEFINE_string(output_path, "results/", "output path for our tests");

void PreProcessIMU(std::vector<IMU> &imus, const cv::Mat &Rci, const cv::Mat &cPic){
    for(IMU& imu:imus)
    {
        cv::Mat acc(3,1,CV_32FC1), gyro(3,1,CV_32FC1);
        for(int j = 0; j<3; j++)
        {
            acc.at<float>(j) = imu.acc[j];
            gyro.at<float>(j) = imu.gyro[j];
        }
        acc = Rci*acc;
        gyro = Rci*gyro;
        cv::Mat skew = (cv::Mat_<float>(3, 3) << 0, -gyro.at<float>(2), gyro.at<float>(1),
                         gyro.at<float>(2), 0, -gyro.at<float>(0),
                        -gyro.at<float>(1), gyro.at<float>(0), 0);
        cv::Mat wwp = skew*skew*cPic;
        acc += wwp;
        for(int j = 0; j<3; j++)
        {
            imu.acc[j] = acc.at<float>(j);
            imu.gyro[j] = gyro.at<float>(j);
        }
    }
}

std::vector<IMU> AverageIMUData(const std::vector<IMU> &imu){
    std::vector<IMU> imuAverage;
    if(imu.empty()) return imuAverage;
    IMU tmpimu;
    memset(&tmpimu, 0, sizeof(IMU));
    //tmpimu.timestamp = 0;
    //memset(static_cast<char*>(tmpimu.acc), 0, sizeof(float)*3)
    //memset(static_cast<char*>(tmpimu.gyro), 0, sizeof(float)*3)
    size_t nimu = imu.size();
    for (size_t i = 0, iend = imu.size(); i<iend; i++){
        tmpimu.timestamp += imu[i].timestamp;
        for (size_t j = 0; j<3; j++){
            tmpimu.acc[j] += imu[i].acc[j];
            tmpimu.gyro[j] += imu[i].gyro[j];
        }
    }
    tmpimu.timestamp = (imu.end()-1)->timestamp;
    for (size_t j = 0; j<3; j++){
        tmpimu.acc[j] /= nimu;
        tmpimu.gyro[j] /= nimu;
    }
    imuAverage.emplace_back(tmpimu);
    return imuAverage;
}


typedef StateD18 StateD;
int main(int argc, char **argv){
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_log_dir = "/tmp/";
    // FLAGS_alsologtostderr = 0;

    ReadGlobalParaFromYaml(FLAGS_yaml_path);
    string g_data_path = FLAGS_data_path;
    int pos = g_data_path.rfind('/')+1;
    string video_name = g_data_path.substr(pos, g_data_path.length()-pos);
    std::cout << video_name << std::endl;

    TimeVideoImuProvider _p_video_imu_provider(g_data_path);
	unsigned char* _buffer = new unsigned char[g_frame_rows*g_frame_cols];
	double _time_stamp;
	float _imu_data[12];
	const cv::Mat &Rci = g_Tci.rowRange(0, 3).colRange(0, 3);
	cv::Mat cPic = -g_Tci.rowRange(0, 3).col(3);
	cv::Mat imLast;
	int nimg = 0;

    StateD::Vector3 p0(0.0, 0.0, 0.0), v0(0., 0., -0.);
    StateD::Quaternion rot0;
    rot0.setIdentity();
    StateD::Vector2 n0(0., -0.);
    double alpha0 = g_init_alpha0;
	int nstate = StateD::stateDim;

    StateD::MatrixIMUNoise stateNoise;
    stateNoise.setIdentity();
    stateNoise.block<3,3>(0, 0) *= g_sigma_pos;
    stateNoise.block<3,3>(3, 3) *= g_sigma_vel;
    stateNoise.block<2,2>(6, 6) *= g_sigma_att;
    stateNoise(8, 8) = g_sigma_yaw;
    stateNoise.block<3,3>(9, 9) *= g_sigma_ba;
    stateNoise.block<3,3>(12, 12) *= g_sigma_bw;
    stateNoise(15, 15) *= g_sigma_alpha;
    stateNoise(16, 16) *= g_sigma_n;
    stateNoise(17, 17) *= g_sigma_n;

    StateD::MatrixState P0;
	P0.setIdentity();
    P0.block<3,3>(0, 0) *= g_sigma_pos0;
    P0.block<3,3>(3, 3) *= g_sigma_vel0;
    P0.block<3,3>(6, 6) *= g_sigma_att0;
    P0.block<3,3>(9, 9) *= g_sigma_ba0;
    P0.block<3,3>(12, 12) *= g_sigma_bw0;
    P0(15, 15) *= g_sigma_alpha0;
    P0(16, 16) *= g_sigma_n0;
    P0(17, 17) *= g_sigma_n0;

    StateD *state;
    state = new StateD(alpha0, p0, v0, rot0, n0, stateNoise, P0);

	float lastTime;

	char est_buf[256], n_buf[256];
	sprintf(est_buf, "%s/%s_est.txt", FLAGS_output_path.c_str(), video_name.c_str());
	ofstream fest(est_buf);
    fest << "#time px py pz qw qx qy qz timecost vx vy vz nx ny nz d bax bay baz bwx bwy bwz wx wy wz" << std::endl;
    LOG(INFO) << "Tci: " << g_Tci;
	while(true){
		float float_cur_time_stamp = 0.0;
		std::vector<IMU> imu_raw_data;
		bool b_load = _p_video_imu_provider.get_image_imu(_buffer, _imu_data, _time_stamp
		, float_cur_time_stamp, imu_raw_data
		);
		if(!b_load || (g_nframes > 0 && nimg > g_nframes)) break;
		if(imu_raw_data.empty()) {
		    LOG(INFO) << "imu empty";
		    continue;
		}
		cv::Mat im(g_frame_rows, g_frame_cols, CV_8UC1, _buffer);
		if(nimg++ == 0)
		{
            imLast = im.clone();
			lastTime = float_cur_time_stamp;
			state->mfCurentTime = float_cur_time_stamp;
			continue;
		}
		PreProcessIMU(imu_raw_data, Rci, cPic);
		for(int i = 0, iend = imu_raw_data.size(); i<iend; i++){
		    IMU & imu = imu_raw_data[i];
			LOG(INFO) << boost::format("%f %f %f %f %f %f %f %f\n") % float_cur_time_stamp % imu.timestamp % imu.acc[0] % imu.acc[1] % imu.acc[2] % imu.gyro[0] % imu.gyro[1] % imu.gyro[2];
		}
        state->PropagationIMUVIO(imu_raw_data);
		START_CV_TIME(tMeasurementUpdate);
        if (g_use_lk)
            state->MeasurementUpdateKFLastLK(imu_raw_data, imLast, im, float_cur_time_stamp - lastTime);
        else{
            state->MeasurementUpdateKFLastMarginalization(imu_raw_data, imLast, im, float_cur_time_stamp - lastTime);
        }
        END_CV_TIME_MS(tMeasurementUpdate);
		const StateD::Vector3& pos = state->mPos;
        const StateD::Vector3& vel = state->mVel;
		const StateD::Quaternion& rot = state->mRot;
        float d = 1.f/state->mfAlpha;
        StateD::Vector3 n;
        n = state->mUnitDirection.getVec();
        const StateD::Vector3 &ba = state->mBa;
        const StateD::Vector3 &bw = state->mBw;
        std::vector<IMU> imus = AverageIMUData(imu_raw_data);
		fest << std::fixed << std::setprecision(6) << _time_stamp << " " << pos(0) << " " << pos(1) << " " << pos(2)
		<< " " << rot.w() << " " << rot.x() << " " << rot.y() << " " << rot.z() << " " << tMeasurementUpdate
        << " " << vel(0) << " " << vel(1) << " " << vel(2)
        << " " << n(0) << " " << n(1) << " " << n(2) << " " << d 
        << " " << ba(0) << " " << ba(1) << " " << ba(2) 
        << " " << bw(0) << " " << bw(1) << " " << bw(2)
        << " " << imus[0].gyro[0] << " " << imus[0].gyro[1] << " " << imus[0].gyro[2]
		<< std::endl;
		imLast = im.clone();
		lastTime = float_cur_time_stamp;
        LOG(INFO) << "ba = " << state->mBa.transpose() << " bw = " << state->mBw.transpose();
	}
	delete _buffer;
	delete state;

	return 0;
}
