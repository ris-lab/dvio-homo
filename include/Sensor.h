//
// Created by zsk on 18-11-25.
//

#ifndef OF_VELOCITY_SENSOR_H
#define OF_VELOCITY_SENSOR_H
#define GRAVITY (9.81f)
struct IMU {
    float acc[3];
    float gyro[3];
	float rot[9];
    float timestamp;
};

struct EurocGroundTruth{
	float timestamp;
	float q[4];
	float p[3];
};
struct CompareGroundTruth{
	bool operator()(const EurocGroundTruth& gt1, const EurocGroundTruth &gt2){
		return gt1.timestamp < gt2.timestamp;
	}

	bool operator()(float t1, const EurocGroundTruth &gt2){
		return t1 < gt2.timestamp;
	}

	bool operator()(const EurocGroundTruth& gt1, float t2){
		return gt1.timestamp < t2;
	}
};
#endif //OF_VELOCITY_SENSOR_H
