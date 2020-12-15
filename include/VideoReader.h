#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <glog/logging.h>
#include "setting_config.h"

#include "Sensor.h"
#include <iomanip>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/lexical_cast.hpp>

class TimeVideoImuProvider {
public:
	TimeVideoImuProvider(const std::string &data_path):
	_binit(false), _nimg(0), _init_time_stamp(false){

        std::string imu_raw_path = data_path + "/imu0.csv";
        LOG(INFO) << "imu_raw_path: " << imu_raw_path;
        _imu_raw_infile.open(imu_raw_path);
        if(!_imu_raw_infile.is_open())
        {
            LOG(FATAL) << "imu file does not exist" << std::endl;
            return;
        }

        std::string img_path;
        img_path = data_path + "/left";

        if (!boost::filesystem::is_directory(img_path)) {
            LOG(ERROR) << " Check the image path";
            return;
        }
        _time_scale = 1e-6;

		std::vector<boost::filesystem::path> img_file_names;  // so we can sort them later
		std::copy(boost::filesystem::directory_iterator(img_path), boost::filesystem::directory_iterator(),
				std::back_inserter(img_file_names));

		for (auto t : img_file_names) {
			std::string str = t.string();
			int idx = str.rfind('.')+1;
			std::string ext = str.substr(idx, str.length() - idx);
			if(ext != "png" && ext != "jpg")
				continue;
			_images_name.push_back(str);
		}
		std::sort(_images_name.begin(), _images_name.end(), [](const std::string &a, std::string &b){
				int aidx1 = a.rfind('/')+1;
				int aidx2 = a.rfind('.');
                uint64_t name1 = atoll(a.substr(aidx1, aidx2-aidx1).c_str());
				int bidx1 = b.rfind('/')+1;
				int bidx2 = b.rfind('.');
                uint64_t name2 = atoll(b.substr(bidx1, bidx2-bidx1).c_str());
				return name1 <= name2;
				});

		LOG(INFO) << "load img size:" << _images_name.size();
	}

    ~TimeVideoImuProvider() {
        if (_imu_raw_infile.is_open()) {
            _imu_raw_infile.close();
        }
    };


    bool get_image_imu(unsigned char* _img_buffer, float* imu_data, double& time_stamp
                       , float& float_time_stamp, std::vector<IMU>& imu_raw_data
                      ) {
        time_stamp = 0;
		double cur_time_stamp = 0;

        if(_nimg >= _images_name.size())
            return false;
        // discard img data before first imu
        std::string tmpstr;
        if (!_binit){
            do{
                getline(_imu_raw_infile, tmpstr, '\n');
            }while(!(tmpstr[0] >= '0' && tmpstr[0] < '9') && tmpstr[0] != '.');

            int pos = tmpstr.find(',');
            double first_imu_t;
            first_imu_t = atof(tmpstr.substr(0, pos).c_str())*_time_scale;
            do{
                const std::string &impath = _images_name[_nimg];
                frame = cv::imread(impath);
                int64_t real_time;
                cur_time_stamp = get_timestamp_from_img_name(impath, real_time);
                cur_time_stamp = cur_time_stamp + g_time_shift;
                if(frame.empty() || _nimg >= _images_name.size())
                    return false;
                _nimg++;
                if(cur_time_stamp > first_imu_t+0.03){
                    LOG(INFO) << "cur time: " << cur_time_stamp;
                    _binit = true;
                    LOG(INFO) << "set true";
                }
            }while(!_binit);
        }else{
            if(frame.empty() || _nimg >= _images_name.size())
                return false;
            const std::string &impath = _images_name[_nimg];
            frame = cv::imread(impath);
            LOG(INFO) << "img path: " << impath;
            int64_t real_time;
            cur_time_stamp = get_timestamp_from_img_name(impath, real_time);
            cur_time_stamp = cur_time_stamp + g_time_shift;
            _nimg++;
        }

		if(frame.channels() == 3)
#if CV_VERSION_MAJOR == 4
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
#else
			cv::cvtColor(frame, gray, CV_BGR2GRAY);
#endif
		else if(frame.channels() == 1)
			gray = frame;

        if (gray.rows != g_frame_rows || gray.cols != g_frame_cols) {
            cv::resize(gray, gray, cv::Size(g_frame_cols, g_frame_rows));
        }

        memcpy(_img_buffer, (unsigned char*) gray.data,
               sizeof(unsigned char) * gray.rows * gray.cols);
		cur_time_stamp += g_time_shift;

        bool imueof = get_raw_imu_data(imu_raw_data, cur_time_stamp, float_time_stamp);

        time_stamp = cur_time_stamp;

        return imueof;
    }

	double get_timestamp_from_img_name(const std::string& img_name, int64_t& real_time) 
	{
		std::string ts_string = boost::filesystem::path(img_name).stem().string();
		int64_t t = boost::lexical_cast<uint64_t>(ts_string);
		real_time = t;
		return double(t) * _time_scale;
	}

    bool get_raw_imu_data(std::vector<IMU>& imu_raw_data,
                     double cur_time_stamp,
                     float& float_time_stamp) {

        if (!_last_imu_raw_data.empty()) {
            if (_last_imu_raw_data[0] + _start_time_stamp > cur_time_stamp) {
                //printf("%f\n", _last_imu_raw_data[0]);
                LOG(INFO) << "last imu time > cur image time";
                return true;
            }

            imu_raw_data.push_back(imu_from_vec(_last_imu_raw_data));
            _last_imu_raw_data.clear();
        }

		const int jump = 1;
		int lines = 1;
        while (true) {
            if (_imu_raw_infile.eof()) {
                return false;
            }

			std::string tmp;
			std::vector<float> tmpdata;
            getline(_imu_raw_infile, tmp, '\n');
            if (tmp.empty())
                break;
            std::stringstream ss(tmp);
            std::string token;
            char delim = ',';
            std::getline(ss, token, delim);
            double timestamp = atof(token.c_str())*_time_scale;
            if (!_init_time_stamp) {
                _start_time_stamp = cur_time_stamp;
                _init_time_stamp = true;
            }
            float pro_time_stamp = timestamp - _start_time_stamp;
            tmpdata.push_back(pro_time_stamp);

            while (std::getline(ss, token, delim)) {
                tmpdata.push_back(atof(token.c_str()));
            }

            if (cur_time_stamp <= timestamp ) {
                _last_imu_raw_data.assign(tmpdata.begin(), tmpdata.end());
                break;
            }
            imu_raw_data.push_back(imu_from_vec(tmpdata));
        }

        float_time_stamp = cur_time_stamp - _start_time_stamp;
        return true;
    }

    inline IMU imu_from_vec(const std::vector<float>& raw) {
        IMU imu;
        imu.timestamp = raw[0];
        std::copy_n(&raw[1], 3, imu.acc);
        std::copy_n(&raw[4], 3, imu.gyro);
        return imu;
    }


private:
    std::ifstream _imu_raw_infile;
    double _time_scale;
    cv::Mat frame;
    cv::Mat gray;
    std::vector<float> _last_imu_raw_data;
    double _start_time_stamp;
    bool _init_time_stamp;
	std::vector<std::string> _images_name;
	int _nimg;
	bool _binit;
};

