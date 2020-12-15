//
// Created by zsk on 19-5-1.
//

#ifndef OF_VELOCITY_NORMALVECTORELEMENT_H
#define OF_VELOCITY_NORMALVECTORELEMENT_H
#include <kindr/Core>
using namespace std;


template<typename Scalar>
class NormalVectorElement{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef kindr::RotationQuaternion<Scalar> QPD;
    typedef kindr::RotationMatrix<Scalar> MPD;
    typedef Eigen::Matrix<Scalar, 3, 1> V3D;
    typedef Eigen::Matrix<Scalar, 2, 1> V2D;
    typedef Eigen::Matrix<Scalar, 3, 3> M3D;
    typedef Eigen::Matrix<Scalar, 2, 2> M2D;



    static M3D gSM(const V3D& vec){
        return kindr::getSkewMatrixFromVector(vec);
    }

    static M3D Lmat (const V3D& a) {
        return kindr::getJacobianOfExponentialMap(a);
    }

    QPD q_;
    const V3D e_x;
    const V3D e_y;
    const V3D e_z;
    NormalVectorElement(): e_x(1,0,0), e_y(0,1,0), e_z(0,0,1){}
    NormalVectorElement(const NormalVectorElement& other): e_x(1,0,0), e_y(0,1,0), e_z(0,0,1){
        q_ = other.q_;
    }
    NormalVectorElement(const V3D& vec): e_x(1,0,0), e_y(0,1,0), e_z(0,0,1){
        setFromVector(vec);
    }
    NormalVectorElement(const QPD& q): e_x(1,0,0), e_y(0,1,0), e_z(0,0,1){
        q_ = q;
    }
    virtual ~NormalVectorElement(){};
    V3D getVec() const{
        return q_.rotate(e_z);
    }
    V3D getPerp1() const{
        return q_.rotate(e_x);
    }
    V3D getPerp2() const{
        return q_.rotate(e_y);
    }
    NormalVectorElement& operator=(const NormalVectorElement& other){
        q_ = other.q_;
        return *this;
    }
    static V3D getRotationFromTwoNormals(const V3D& a, const V3D& b, const V3D& a_perp){
        const V3D cross = a.cross(b);
        const Scalar crossNorm = cross.norm();
        const Scalar c = a.dot(b);
        const Scalar angle = std::acos(c);
        if(crossNorm<1e-2){
            if(c>0){
                return cross;
            } else {
                return a_perp*M_PI;
            }
        } else {
            return cross*(angle/crossNorm);
        }
    }
    static V3D getRotationFromTwoNormals(const NormalVectorElement& a, const NormalVectorElement& b){
        return getRotationFromTwoNormals(a.getVec(),b.getVec(),a.getPerp1());
    }
    static M3D getRotationFromTwoNormalsJac(const V3D& a, const V3D& b){
        const V3D cross = a.cross(b);
        const Scalar crossNorm = cross.norm();
        V3D crossNormalized = cross/crossNorm;
        M3D crossNormalizedSqew = gSM(crossNormalized);
        const Scalar c = a.dot(b);
        const Scalar angle = std::acos(c);
        if(crossNorm<1e-6){
            if(c>0){
                return -gSM(b);
            } else {
                return M3D::Zero();
            }
        } else {
            return -1/crossNorm*(crossNormalized*b.transpose()-(crossNormalizedSqew*crossNormalizedSqew*gSM(b)*angle));
        }
    }
    static M3D getRotationFromTwoNormalsJac(const NormalVectorElement& a, const NormalVectorElement& b){
        return getRotationFromTwoNormalsJac(a.getVec(),b.getVec());
    }
    void setFromVector(V3D vec){
        const Scalar d = vec.norm();
        if(d > 1e-6){
            vec = vec/d;
            q_ = q_.exponentialMap(getRotationFromTwoNormals(e_z,vec,e_x));
        } else {
            q_.setIdentity();
        }
    }
    NormalVectorElement rotated(const QPD& q) const{
        return NormalVectorElement(q*q_);
    }
    NormalVectorElement inverted() const{
        QPD q = q.exponentialMap(M_PI*getPerp1());
        return NormalVectorElement(q*q_);
    }
    void boxPlus(const V2D& vecIn, NormalVectorElement& stateOut) const{
        QPD q = q.exponentialMap(vecIn(0)*getPerp1()+vecIn(1)*getPerp2());
        stateOut.q_ = q*q_;
    }
    void boxPlus(const V2D& vecIn) {
        QPD q = q.exponentialMap(vecIn(0)*getPerp1()+vecIn(1)*getPerp2());
        q_ = q*q_;
    }
    // stateIn boxminus this
    // cross(statein, this) has a negative direction from the IJRR
    // the negative direction comes from the sign of C(exp)
    // the jacobian of cross(a, b) should be negative
    void boxMinus(const NormalVectorElement& stateIn, V2D& vecOut) const{
        vecOut = stateIn.getN().transpose()*getRotationFromTwoNormals(stateIn,*this);
    }
    // jacobian of stateIn boxminus this
    void boxMinusJac(const NormalVectorElement& stateIn, M2D& matOut) const{
        matOut = -stateIn.getN().transpose()*getRotationFromTwoNormalsJac(*this,stateIn)*this->getM();
    }
    void print() const{
        std::cout << getVec().transpose() << std::endl;
    }
    void setIdentity(){
        q_.setIdentity();
    }
    void setRandom(unsigned int& s){
        std::default_random_engine generator (s);
        std::normal_distribution<Scalar> distribution (0.0,1.0);
        q_.toImplementation().w() = distribution(generator);
        q_.toImplementation().x() = distribution(generator);
        q_.toImplementation().y() = distribution(generator);
        q_.toImplementation().z() = distribution(generator);
        q_.fix();
        s++;
    }
    void fix(){
        q_.fix();
    }
//    mtGet& get(unsigned int i = 0){
//        assert(i==0);
//        return *this;
//    }
//    const mtGet& get(unsigned int i = 0) const{
//        assert(i==0);
//        return *this;
//    }
//    void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){
//        mpPropertyHandler->ScalarRegister_.registerQuaternion(str + name_, q_);
//    }
    Eigen::Matrix<Scalar,3,2> getM() const {
        Eigen::Matrix<Scalar,3,2> M;
        M.col(0) = -getPerp2();
        M.col(1) = getPerp1();
        return M;
    }
    Eigen::Matrix<Scalar,3,2> getN() const {
        Eigen::Matrix<Scalar,3,2> M;
        M.col(0) = getPerp1();
        M.col(1) = getPerp2();
        return M;
    }
};
typedef NormalVectorElement<float> NormalVectorElementF;
typedef NormalVectorElement<double> NormalVectorElementD;

#endif //OF_VELOCITY_NORMALVECTORELEMENT_H
