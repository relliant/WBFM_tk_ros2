#include <Eigen/Dense>

#include "third_party/funcSPTrans.h"

extern "C" {

void* funcsptrans_create() {
    return new funcSPTrans();
}

void funcsptrans_destroy(void* handle) {
    delete static_cast<funcSPTrans*>(handle);
}

int funcsptrans_state_to_joint(
    void* handle,
    const double* q_p,
    const double* qdot_p,
    const double* tor_p,
    double* q_s,
    double* qdot_s,
    double* tor_s
) {
    if (handle == nullptr) {
        return -1;
    }

    try {
        auto* transmission = static_cast<funcSPTrans*>(handle);
        Eigen::VectorXd qPEst = Eigen::Map<const Eigen::VectorXd>(q_p, 4);
        Eigen::VectorXd qDotPEst = Eigen::Map<const Eigen::VectorXd>(qdot_p, 4);
        Eigen::VectorXd qTorPEst = Eigen::Map<const Eigen::VectorXd>(tor_p, 4);
        Eigen::VectorXd qSEst(4);
        Eigen::VectorXd qDotSEst(4);
        Eigen::VectorXd torSEst(4);

        transmission->setPEst(qPEst, qDotPEst, qTorPEst);
        transmission->calcFK();
        transmission->calcIK();
        transmission->getSState(qSEst, qDotSEst, torSEst);

        Eigen::Map<Eigen::VectorXd>(q_s, 4) = qSEst;
        Eigen::Map<Eigen::VectorXd>(qdot_s, 4) = qDotSEst;
        Eigen::Map<Eigen::VectorXd>(tor_s, 4) = torSEst;
        return 0;
    } catch (...) {
        return -2;
    }
}

int funcsptrans_joint_to_motor(
    void* handle,
    const double* q_s,
    const double* qdot_s,
    const double* tor_s,
    double* q_p,
    double* qdot_p,
    double* tor_p
) {
    if (handle == nullptr) {
        return -1;
    }

    try {
        auto* transmission = static_cast<funcSPTrans*>(handle);
        Eigen::VectorXd qSRef = Eigen::Map<const Eigen::VectorXd>(q_s, 4);
        Eigen::VectorXd qDotSRef = Eigen::Map<const Eigen::VectorXd>(qdot_s, 4);
        Eigen::VectorXd torSDes = Eigen::Map<const Eigen::VectorXd>(tor_s, 4);
        Eigen::VectorXd qPDes(4);
        Eigen::VectorXd qDotPDes(4);
        Eigen::VectorXd torPDes(4);

        transmission->setSDes(qSRef, qDotSRef, torSDes);
        transmission->calcJointPosRef();
        transmission->calcJointTorDes();
        transmission->getPDes(qPDes, qDotPDes, torPDes);

        Eigen::Map<Eigen::VectorXd>(q_p, 4) = qPDes;
        Eigen::Map<Eigen::VectorXd>(qdot_p, 4) = qDotPDes;
        Eigen::Map<Eigen::VectorXd>(tor_p, 4) = torPDes;
        return 0;
    } catch (...) {
        return -2;
    }
}

}  // extern "C"
