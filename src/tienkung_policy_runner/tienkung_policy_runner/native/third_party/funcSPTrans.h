#ifndef S2P_H_
#define S2P_H_

#ifdef _WIN32
  #ifdef FUNCSPTRANS_EXPORTS
    #define FUNCSPTRANS_API __declspec(dllexport)
  #else
    #define FUNCSPTRANS_API __declspec(dllimport)
  #endif
#else
  #define FUNCSPTRANS_API
#endif

#ifndef PI
#define PI 3.141592654
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

#include <Eigen/Dense>

class FUNCSPTRANS_API funcSPTrans {
 public:
  funcSPTrans();
  ~funcSPTrans();

  bool calcJointPosRef();
  bool calcJointPosEst();
  bool calcJointPosRefLeft();
  bool calcJointPosRefRight();
  bool calcJointPosEstLeft();
  bool calcJointPosEstRight();
  bool calcIK();
  bool calcJLeft();
  bool calcJRight();
  bool calcFK();
  bool calcAnkleEstLeft();
  bool calcAnkleEstRight();
  bool calcJointTorDes();
  bool setPEst(const Eigen::VectorXd &qPEst,
               const Eigen::VectorXd &qDotPEst,
               const Eigen::VectorXd &qTorPEst);

  bool getSState(Eigen::VectorXd &qSEst,
                 Eigen::VectorXd &qDotSEst,
                 Eigen::VectorXd &torSEst);
  bool setSState(Eigen::VectorXd &qSEst);

  bool setSDes(const Eigen::VectorXd &qSRef,
               const Eigen::VectorXd &qDotSRef,
               const Eigen::VectorXd &torSDes);

  bool getPDes(Eigen::VectorXd &qPDes,
               Eigen::VectorXd &qDotPDes,
               Eigen::VectorXd &torPDes);

  Eigen::Matrix3d Skew(const Eigen::Vector3d &omg);

  double BC_l;
  double OA1_l;
  double OA2_l;
  double C1P1_l;
  double C2P2_l;
  double AB_l;
  double BC_r;
  double OA1_r;
  double OA2_r;
  double C1P1_r;
  double C2P2_r;
  double AB_r;

  Eigen::MatrixXd rotXLeft;
  Eigen::MatrixXd rotYLeft;
  Eigen::MatrixXd rotYXLeft;
  Eigen::MatrixXd rotXLeftRef;
  Eigen::MatrixXd rotYLeftRef;
  Eigen::MatrixXd rotYXLeftRef;
  Eigen::MatrixXd rotXRight;
  Eigen::MatrixXd rotYRight;
  Eigen::MatrixXd rotYXRight;
  Eigen::MatrixXd rotXRightRef;
  Eigen::MatrixXd rotYRightRef;
  Eigen::MatrixXd rotYXRightRef;

  Eigen::Vector3d OP1_l;
  Eigen::Vector3d OP2_l;
  Eigen::Vector3d OP1_r;
  Eigen::Vector3d OP2_r;

  Eigen::Vector3d oP1BodyLeft;
  Eigen::Vector3d oP2BodyLeft;
  Eigen::Vector3d oP1BodyLeftRef;
  Eigen::Vector3d oP2BodyLeftRef;

  Eigen::Vector3d oP1BodyRight;
  Eigen::Vector3d oP2BodyRight;
  Eigen::Vector3d oP1BodyRightRef;
  Eigen::Vector3d oP2BodyRightRef;

  double roll_l_ref;
  double pitch_l_ref;
  double alpha1_l_ref;
  double alpha2_l_ref;
  Eigen::Vector2d qP_l_ref;
  Eigen::Vector2d qDotP_l_ref;
  Eigen::Vector2d torP_l_ref;
  Eigen::Vector2d qS_l_ref;
  Eigen::Vector2d qDotS_l_ref;
  Eigen::Vector2d torS_l_ref;

  double roll_r_ref;
  double pitch_r_ref;
  double alpha1_r_ref;
  double alpha2_r_ref;
  Eigen::Vector2d qP_r_ref;
  Eigen::Vector2d qDotP_r_ref;
  Eigen::Vector2d torP_r_ref;
  Eigen::Vector2d qS_r_ref;
  Eigen::Vector2d qDotS_r_ref;
  Eigen::Vector2d torS_r_ref;

  double roll_l_est;
  double pitch_l_est;
  double alpha1_l_est;
  double alpha2_l_est;
  Eigen::Vector2d qP_l_est;
  Eigen::Vector2d qDotP_l_est;
  Eigen::Vector2d torP_l_est;
  Eigen::Vector2d qS_l_est;
  Eigen::Vector2d qDotS_l_est;
  Eigen::Vector2d torS_l_est;

  double roll_r_est;
  double pitch_r_est;
  double alpha1_r_est;
  double alpha2_r_est;
  Eigen::Vector2d qP_r_est;
  Eigen::Vector2d qDotP_r_est;
  Eigen::Vector2d torP_r_est;
  Eigen::Vector2d qS_r_est;
  Eigen::Vector2d qDotS_r_est;
  Eigen::Vector2d torS_r_est;

  Eigen::Vector3d C1P1Left;
  Eigen::Vector3d C2P2Left;
  Eigen::Vector3d C1P1Right;
  Eigen::Vector3d C2P2Right;

  Eigen::MatrixXd ROmegaLeft;
  Eigen::MatrixXd ROmegaLegLeft;
  Eigen::MatrixXd JP1Left;
  Eigen::MatrixXd JP2Left;

  Eigen::MatrixXd ROmegaRight;
  Eigen::MatrixXd ROmegaLegRight;
  Eigen::MatrixXd JP1Right;
  Eigen::MatrixXd JP2Right;

  Eigen::Vector3d B1C1Left;
  Eigen::Vector3d B2C2Left;
  Eigen::MatrixXd JLeft;
  Eigen::MatrixXd JAnkleLeft;
  Eigen::MatrixXd JC1Left;
  Eigen::MatrixXd JC2Left;
  Eigen::Vector3d vP1Left;
  Eigen::Vector3d vP2Left;

  Eigen::Vector3d B1C1Right;
  Eigen::Vector3d B2C2Right;
  Eigen::MatrixXd JRight;
  Eigen::MatrixXd JAnkleRight;
  Eigen::MatrixXd JC1Right;
  Eigen::MatrixXd JC2Right;
  Eigen::Vector3d vP1Right;
  Eigen::Vector3d vP2Right;

  Eigen::Vector4d tauDesjointFB;
};

#endif
