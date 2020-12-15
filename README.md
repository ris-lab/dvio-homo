# An Efficient Iterated EKF-based Direct Visual-Inertial Odometry for MAVs Using a Single Plane Primitive 
By Shangkun Zhong and Pakpong Chirarattananon

## Introduction

As an incremental [work](https://arxiv.org/abs/2001.05215), this letter proposes an efficient visual-inertial estimator for aerial robots.  For more details, please refer to our RA-L [paper]().

## Dependencies
c++11, opencv, eigen, kindr, boost, glog, gflags

## Getting started
### Compilation
Please run the following commands to compile the code.
```bash
mkdir build
cd build
cmake ..
make
```
### Run
The collected datasets in the paper can be downloaded [here](https://portland-my.sharepoint.com/personal/shanzhong4-c_my_cityu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshanzhong4%2Dc%5Fmy%5Fcityu%5Fedu%5Fhk%2FDocuments%2Fdatasets%2Fpublic). To run the algorithm with the datasets, for example, extract the seq1 (1.zip) 
```
unzip 1.zip
``` 
and run:
```
./build/main --yaml_path bebop_bottom.yaml --data_path seq1_dir/1 --output_path results/
```


## Contact

If you have any questions, please contact me at 291790832@qq.com.
