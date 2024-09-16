//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ComputeIncrementalCurvedBeamStrain.h"
#include "MooseMesh.h"
#include "Assembly.h"
#include "NonlinearSystem.h"
#include "MooseVariable.h"
#include "Function.h"

#include "libmesh/quadrature.h"
#include "libmesh/utility.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_type.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/quadrature_gauss.h"

registerMooseObject(MOOSEAPPNAME, ComputeIncrementalCurvedBeamStrain);

InputParameters
ComputeIncrementalCurvedBeamStrain::validParams()
{
  InputParameters params = Material::validParams();
  params.addClassDescription("Compute a infinitesimal/large strain increment for the beam.");
  params.addRequiredCoupledVar(
      "rotations", "The rotations appropriate for the simulation geometry and coordinate system");
  params.addRequiredCoupledVar(
      "displacements",
      "The displacements appropriate for the simulation geometry and coordinate system");
  params.addParam<RealGradient>("y_orientation",
                                        "Orientation of the y direction along "
                                        "with Iyy is provided. This should be "
                                        "perpendicular to the axis of the beam.");
  params.addRequiredCoupledVar(
      "area",
      "Cross-section area of the beam. Can be supplied as either a number or a variable name.");
  params.addCoupledVar("Ay",
                       0.0,
                       "First moment of area of the beam about y axis. Can be supplied "
                       "as either a number or a variable name.");
  params.addCoupledVar("Az",
                       0.0,
                       "First moment of area of the beam about z axis. Can be supplied "
                       "as either a number or a variable name.");
  params.addCoupledVar("Ix",
                       "Second moment of area of the beam about x axis. Can be "
                       "supplied as either a number or a variable name. Defaults to Iy+Iz.");
  params.addRequiredCoupledVar("Iy",
                               "Second moment of area of the beam about y axis. Can be "
                               "supplied as either a number or a variable name.");
  params.addRequiredCoupledVar("Iz",
                               "Second moment of area of the beam about z axis. Can be "
                               "supplied as either a number or a variable name.");
  params.addParam<bool>("large_strain", false, "Set to true if large strain are to be calculated.");
  params.addParam<std::vector<MaterialPropertyName>>(
      "eigenstrain_names",
      {},
      "List of beam eigenstrains to be applied in this strain calculation.");
  params.addParam<FunctionName>(
      "elasticity_prefactor",
      "Optional function to use as a scalar prefactor on the elasticity vector for the beam.");
  return params;
}

ComputeIncrementalCurvedBeamStrain::ComputeIncrementalCurvedBeamStrain(const InputParameters & parameters)
  : Material(parameters),
    _has_Ix(isParamValid("Ix")),
    _nrot(coupledComponents("rotations")),
    _ndisp(coupledComponents("displacements")),
    _rot_num(_nrot),
    _disp_num(_ndisp),
    _area(coupledValue("area")),
    _Ay(coupledValue("Ay")),
    _Az(coupledValue("Az")),
    _Iy(coupledValue("Iy")),
    _Iz(coupledValue("Iz")),
    _Ix(_has_Ix ? coupledValue("Ix") : _zero),
    _original_length(declareProperty<Real>("original_length")),
    _total_rotation(declareProperty<RankTwoTensor>("total_rotation")),
    _total_disp_strain(declareProperty<RealVectorValue>("total_disp_strain")),
    _total_rot_strain(declareProperty<RealVectorValue>("total_rot_strain")),
    _total_disp_strain_old(getMaterialPropertyOld<RealVectorValue>("total_disp_strain")),
    _total_rot_strain_old(getMaterialPropertyOld<RealVectorValue>("total_rot_strain")),
    _mech_disp_strain_increment(declareProperty<RealVectorValue>("mech_disp_strain_increment")),
    _mech_rot_strain_increment(declareProperty<RealVectorValue>("mech_rot_strain_increment")),
    _material_stiffness(getMaterialPropertyByName<RealVectorValue>("material_stiffness")),
    _Kuu00(declareProperty<RankTwoTensor>("Jacobian_00")),
    _Kut01(declareProperty<RankTwoTensor>("Jacobian_01")),
    _Kuu02(declareProperty<RankTwoTensor>("Jacobian_02")),
    _Kut03(declareProperty<RankTwoTensor>("Jacobian_03")),
    _Kuu04(declareProperty<RankTwoTensor>("Jacobian_04")),
    _Kut05(declareProperty<RankTwoTensor>("Jacobian_05")),
    _Ktt11(declareProperty<RankTwoTensor>("Jacobian_11")),
    _Ktu12(declareProperty<RankTwoTensor>("Jacobian_12")),
    _Ktt13(declareProperty<RankTwoTensor>("Jacobian_13")),
    _Ktu14(declareProperty<RankTwoTensor>("Jacobian_14")),
    _Ktt15(declareProperty<RankTwoTensor>("Jacobian_15")),
    _Kuu22(declareProperty<RankTwoTensor>("Jacobian_22")),
    _Kut23(declareProperty<RankTwoTensor>("Jacobian_23")),
    _Kuu24(declareProperty<RankTwoTensor>("Jacobian_24")),
    _Kut25(declareProperty<RankTwoTensor>("Jacobian_25")),
    _Ktt33(declareProperty<RankTwoTensor>("Jacobian_33")),
    _Ktu34(declareProperty<RankTwoTensor>("Jacobian_34")),
    _Ktt35(declareProperty<RankTwoTensor>("Jacobian_35")),
    _Kuu44(declareProperty<RankTwoTensor>("Jacobian_44")),
    _Kut45(declareProperty<RankTwoTensor>("Jacobian_45")),
    _Ktt55(declareProperty<RankTwoTensor>("Jacobian_55")),
    _large_strain(getParam<bool>("large_strain")),
    _eigenstrain_names(getParam<std::vector<MaterialPropertyName>>("eigenstrain_names")),
    _disp_eigenstrain(_eigenstrain_names.size()),
    _rot_eigenstrain(_eigenstrain_names.size()),
    _disp_eigenstrain_old(_eigenstrain_names.size()),
    _rot_eigenstrain_old(_eigenstrain_names.size()),
    _nonlinear_sys(_fe_problem.getNonlinearSystemBase(/*nl_sys_num=*/0)),
    _soln_disp_index_0(_ndisp),
    _soln_disp_index_1(_ndisp),
    _soln_disp_index_2(_ndisp),
    _soln_rot_index_0(_ndisp),
    _soln_rot_index_1(_ndisp),
    _soln_rot_index_2(_ndisp),
    _nodes(3),
    _initial_rotation(declareProperty<RankTwoTensor>("initial_rotation")),
    _effective_stiffness(declareProperty<Real>("effective_stiffness")),
    _prefactor_function(isParamValid("elasticity_prefactor") ? &getFunction("elasticity_prefactor")
                                                             : nullptr),
    _has_y_vector(isParamValid("y_orientation"))
{
  // Checking for consistency between length of the provided displacements and rotations vector
  if (_ndisp != _nrot)
    mooseError("ComputeIncrementalCurvedBeamStrain: The number of variables supplied in 'displacements' "
               "and 'rotations' must match.");

  // fetch coupled variables and gradients (as stateful properties if necessary)
  for (unsigned int i = 0; i < _ndisp; ++i)
  {
    MooseVariable * disp_variable = getVar("displacements", i);
    _disp_num[i] = disp_variable->number();

    MooseVariable * rot_variable = getVar("rotations", i);
    _rot_num[i] = rot_variable->number();
  }

  if (_large_strain)
    mooseError("Large strain calculation is not implemented");

  // for (unsigned int i = 0; i < _eigenstrain_names.size(); ++i)
  // {
  //   _disp_eigenstrain[i] = &getMaterialProperty<RealVectorValue>("disp_" + _eigenstrain_names[i]);
  //   _rot_eigenstrain[i] = &getMaterialProperty<RealVectorValue>("rot_" + _eigenstrain_names[i]);
  //   _disp_eigenstrain_old[i] =
  //       &getMaterialPropertyOld<RealVectorValue>("disp_" + _eigenstrain_names[i]);
  //   _rot_eigenstrain_old[i] =
  //       &getMaterialPropertyOld<RealVectorValue>("rot_" + _eigenstrain_names[i]);
  // }
}

void
ComputeIncrementalCurvedBeamStrain::initQpStatefulProperties()
{
  // compute initial orientation of the beam for calculating initial rotation matrix
  FEType fe_type(Utility::string_to_enum<Order>("SECOND"),
                 Utility::string_to_enum<FEFamily>("LAGRANGE"));
 const std::vector<RealGradient> * orientation =&_subproblem.assembly(_tid, _nonlinear_sys.number()).getFE(fe_type, 1)->get_dxyzdxi();
  RealGradient x_orientation=(*orientation)[0];
 
  x_orientation /= x_orientation.norm();
  RealGradient y_orientation={0,1,0};
  RealGradient z_orientation={0,0,1};
  Real sum=0.0;
  if (_has_y_vector)
  {
    y_orientation = getParam<RealGradient>("y_orientation");
  }else{
    y_orientation =z_orientation.cross(x_orientation);
    y_orientation /= y_orientation.norm();
    sum = x_orientation(0) * y_orientation(0) + x_orientation(1) * y_orientation(1) +
             x_orientation(2) * y_orientation(2);
    if (std::abs(sum) > 1e-4)
    {
     y_orientation={0,1,0}; 
    }
  }

  y_orientation /= y_orientation.norm();
  sum = x_orientation(0) * y_orientation(0) + x_orientation(1) * y_orientation(1) +
             x_orientation(2) * y_orientation(2);

  if (std::abs(sum) > 1e-4)
    mooseError("ComputeIncrementalCurvedBeamStrain: y_orientation should be perpendicular to "
               "the axis of the beam.");

  // Calculate z orientation as a cross product of the x and y orientations
  z_orientation=x_orientation.cross(y_orientation);

  // Rotation matrix from global to original beam local configuration
  _original_local_config(0, 0) = x_orientation(0);
  _original_local_config(0, 1) = x_orientation(1);
  _original_local_config(0, 2) = x_orientation(2);
  _original_local_config(1, 0) = y_orientation(0);
  _original_local_config(1, 1) = y_orientation(1);
  _original_local_config(1, 2) = y_orientation(2);
  _original_local_config(2, 0) = z_orientation(0);
  _original_local_config(2, 1) = z_orientation(1);
  _original_local_config(2, 2) = z_orientation(2);

  _total_rotation[_qp] = _original_local_config;

  RealVectorValue temp;
  _total_disp_strain[_qp] = temp;
  _total_rot_strain[_qp] = temp;
}

void
ComputeIncrementalCurvedBeamStrain::computeProperties()
{
  // fetch the two end nodes for current element
  std::vector<const Node *> node;
  for (unsigned int i = 0; i < 3; ++i)
    node.push_back(_current_elem->node_ptr(i));
   
  // calculate original length of a beam element
  // Nodal positions do not change with time as undisplaced mesh is used by material classes by
  // default
  RealGradient dxyz;
  for (unsigned int i = 0; i < _ndisp; ++i)
    dxyz(i) = (*node[1])(i) - (*node[0])(i);

  _original_length[0] = dxyz.norm();

  // Fetch the solution for the two end nodes at time t
  const NumericVector<Number> & sol = *_nonlinear_sys.currentSolution();
  const NumericVector<Number> & sol_old = _nonlinear_sys.solutionOld();

  for (unsigned int i = 0; i < _ndisp; ++i)
  {
    _soln_disp_index_0[i] = node[0]->dof_number(_nonlinear_sys.number(), _disp_num[i], 0);
    _soln_disp_index_1[i] = node[1]->dof_number(_nonlinear_sys.number(), _disp_num[i], 0);
    _soln_disp_index_2[i] = node[2]->dof_number(_nonlinear_sys.number(), _disp_num[i], 0);
    _soln_rot_index_0[i] = node[0]->dof_number(_nonlinear_sys.number(), _rot_num[i], 0);
    _soln_rot_index_1[i] = node[1]->dof_number(_nonlinear_sys.number(), _rot_num[i], 0);
    _soln_rot_index_2[i] = node[2]->dof_number(_nonlinear_sys.number(), _rot_num[i], 0);

    _disp0(i) = sol(_soln_disp_index_0[i]) - sol_old(_soln_disp_index_0[i]);
    _disp1(i) = sol(_soln_disp_index_1[i]) - sol_old(_soln_disp_index_1[i]);
    _disp2(i) = sol(_soln_disp_index_2[i]) - sol_old(_soln_disp_index_2[i]);
    _rot0(i) = sol(_soln_rot_index_0[i]) - sol_old(_soln_rot_index_0[i]);
    _rot1(i) = sol(_soln_rot_index_1[i]) - sol_old(_soln_rot_index_1[i]);
    _rot2(i) = sol(_soln_rot_index_2[i]) - sol_old(_soln_rot_index_2[i]);
   
  }

  // For small rotation problems,cd the rotation matrix is essentially the transformation from the
  // global to original beam local configuration and is never updated. This method has to be
  // overriden for scenarios with finite rotations
  //computeRotation();
  _initial_rotation[0] = _original_local_config;
 
  for (_qp = 0; _qp < _qrule->n_points(); ++_qp)
    computeQpStrain();

  if (_fe_problem.currentlyComputingJacobian())
    computeStiffnessMatrix();
}

void
ComputeIncrementalCurvedBeamStrain::computeQpStrain()
{
  
  Real xi=_qrule->get_points()[_qp](0);
 
  Real h0=0.5*(xi-1.0)*xi;
  Real h1=0.5*(xi+1.0)*xi;
  Real h2=(1.0-xi*xi);
  Real A= (_area[_qp]);
  Real Ay= (_Ay[_qp]);
  Real Az= (_Az[_qp]);
  Real Iz= (_Iz[_qp]);
  Real Iy= (_Iy[_qp]);
  Real Iyz=0.0;

  Real Ix = _Ix[_qp];
  if (!_has_Ix)
    Ix = _Iy[_qp] + _Iz[_qp];

  // Rotate the gradient of displacements and rotations at t+delta t from global coordinate
  // frame to beam local coordinate frame
  RealVectorValue grad_disp_0={(((2.0*xi-1)*_disp0(0) + (2.0*xi+1)*_disp1(0)+(-4.0*xi)*_disp2(0)))/ _original_length[0],
                              (((2.0*xi-1)*_disp0(1) + (2.0*xi+1)*_disp1(1)+(-4.0*xi)*_disp2(1)))/ _original_length[0],
                              (((2.0*xi-1)*_disp0(2) + (2.0*xi+1)*_disp1(2)+(-4.0*xi)*_disp2(2)))/ _original_length[0]};
  RealVectorValue grad_rot_0={(((2.0*xi-1)*_rot0(0) + (2.0*xi+1)*_rot1(0)+(-4.0*xi)*_rot2(0)))/ _original_length[0],
                             (((2.0*xi-1)*_rot0(1) + (2.0*xi+1)*_rot1(1)+(-4.0*xi)*_rot2(1)))/ _original_length[0],
                             (((2.0*xi-1)*_rot0(2) + (2.0*xi+1)*_rot1(2)+(-4.0*xi)*_rot2(2)))/ _original_length[0]};
  RealVectorValue avg_rot={
      (h0*_rot0(0) + h1*_rot1(0)+ h2*_rot2(0)), (h0*_rot0(1) + h1*_rot1(1)+ h2*_rot2(1)), (h0*_rot0(2) + h1*_rot1(2)+ h2*_rot2(2))};

  _grad_disp_0_local_t = _total_rotation[0] * grad_disp_0;
  _grad_rot_0_local_t = _total_rotation[0] * grad_rot_0;
  _avg_rot_local_t = _total_rotation[0] * avg_rot;

  // displacement at any location on beam in local coordinate system at t
  // u_1 = u_n1 - rot_3 * y + rot_2 * z
  // u_2 = u_n2 - rot_1 * z
  // u_3 = u_n3 + rot_1 * y
  // where u_n1, u_n2, u_n3 are displacements at neutral axis

  // small strain
  // e_11 = u_1,1 = u_n1, 1 - rot_3, 1 * y + rot_2, 1 * z
  // e_12 = 2 * 0.5 * (u_1,2 + u_2,1) = (- rot_3 + u_n2,1 - rot_1,1 * z)
  // e_13 = 2 * 0.5 * (u_1,3 + u_3,1) = (rot_2 + u_n3,1 + rot_1,1 * y)

  // axial and shearing strains at each qp along the length of the beam
  _mech_disp_strain_increment[_qp](0) = _grad_disp_0_local_t(0) * _area[_qp] -
                                        _grad_rot_0_local_t(2) * _Ay[_qp] +
                                        _grad_rot_0_local_t(1) * _Az[_qp];
  _mech_disp_strain_increment[_qp](1) = -_avg_rot_local_t(2) * _area[_qp] +
                                        _grad_disp_0_local_t(1) * _area[_qp] -
                                        _grad_rot_0_local_t(0) * _Az[_qp];
  _mech_disp_strain_increment[_qp](2) = _avg_rot_local_t(1) * _area[_qp] +
                                        _grad_disp_0_local_t(2) * _area[_qp] +
                                        _grad_rot_0_local_t(0) * _Ay[_qp];

  // rotational strains at each qp along the length of the beam
  // rot_strain_1 = integral(e_13 * y - e_12 * z) dA
  // rot_strain_2 = integral(e_11 * z) dA
  // rot_strain_3 = integral(e_11 * -y) dA
  // Iyz is the product moment of inertia which is zero for most cross-sections so it is assumed to
  // be zero for this analysis

  _mech_rot_strain_increment[_qp](0) =
      _avg_rot_local_t(1) * _Ay[_qp] + _grad_disp_0_local_t(2) * _Ay[_qp] +
      _grad_rot_0_local_t(0) * Ix + _avg_rot_local_t(2) * _Az[_qp] -
      _grad_disp_0_local_t(1) * _Az[_qp];
  _mech_rot_strain_increment[_qp](1) = _grad_disp_0_local_t(0) * _Az[_qp] -
                                       _grad_rot_0_local_t(2) * Iyz +
                                       _grad_rot_0_local_t(1) * _Iz[_qp];
  _mech_rot_strain_increment[_qp](2) = -_grad_disp_0_local_t(0) * _Ay[_qp] +
                                       _grad_rot_0_local_t(2) * _Iy[_qp] -
                                       _grad_rot_0_local_t(1) * Iyz;

 

  _total_disp_strain[_qp] = _total_rotation[0].transpose() * _mech_disp_strain_increment[_qp] +
                            _total_disp_strain_old[_qp];
  _total_rot_strain[_qp] = _total_rotation[0].transpose() * _mech_rot_strain_increment[_qp] +
                           _total_disp_strain_old[_qp];

  // // Convert eigenstrain increment from global to beam local coordinate system and remove eigen
  // // strain increment
  // for (unsigned int i = 0; i < _eigenstrain_names.size(); ++i)
  // {
  //   _mech_disp_strain_increment[_qp] -=
  //       _total_rotation[0] * ((*_disp_eigenstrain[i])[_qp] - (*_disp_eigenstrain_old[i])[_qp]) *
  //       _area[_qp];
  //   _mech_rot_strain_increment[_qp] -=
  //       _total_rotation[0] * ((*_rot_eigenstrain[i])[_qp] - (*_rot_eigenstrain_old[i])[_qp]);
  // }

  Real c1_paper = std::sqrt(_material_stiffness[0](0));
  Real c2_paper = std::sqrt(_material_stiffness[0](1));

  Real effec_stiff_1 = std::max(c1_paper, c2_paper);

  Real effec_stiff_2 = 2 / (c2_paper * std::sqrt(A / Iz));

  _effective_stiffness[_qp] = std::max(effec_stiff_1, _original_length[0] / effec_stiff_2);

  if (_prefactor_function)
    _effective_stiffness[_qp] *= std::sqrt(_prefactor_function->value(_t, _q_point[_qp]));
}

void
ComputeIncrementalCurvedBeamStrain::computeStiffnessMatrix()
{
  const Real youngs_modulus = _material_stiffness[0](0);
  const Real shear_modulus = _material_stiffness[0](1);

  Real A= (_area[_qp]);
  Real Ay= (_Ay[_qp]);
  Real Az= (_Az[_qp]);
  Real Iz= (_Iz[_qp]);
  Real Iy= (_Iy[_qp]);
  Real Izy=0.0;

  Real Ix = _Ix[_qp];
  if (!_has_Ix)
    Ix = _Iy[_qp] + _Iz[_qp];

  // K = |K11 K12|
  //     |K21 K22|

  // relation between translational displacements at node 0 and translational forces at node 0
  RankTwoTensor K00_local;
  K00_local.zero();
  K00_local(0, 0) = (7.0/3.0)*youngs_modulus * A / _original_length[0];
  K00_local(1, 1) = (7.0/3.0)*shear_modulus * A/ _original_length[0];
  K00_local(2, 2) = (7.0/3.0)*shear_modulus * A/ _original_length[0];
  _Kuu00[0] = _total_rotation[0].transpose() * K00_local * _total_rotation[0];

  // relation between displacements at node 0 and rotational moments at node 0
  RankTwoTensor K01_local;
  K01_local.zero();
  K01_local(0, 1) = (7.0/3.0) * Az* youngs_modulus/_original_length[0];
  K01_local(0, 2) = -(7.0/3.0) * Ay* youngs_modulus/_original_length[0];
  K01_local(1, 0) = -(7.0/3.0) * Az* shear_modulus/_original_length[0];
  K01_local(1, 2) = shear_modulus * A * 0.5;
  K01_local(2, 0) = (7.0/3.0) * Ay* shear_modulus/_original_length[0];
  K01_local(2, 1) = -shear_modulus * A * 0.5;
  _Kut01[0] = _total_rotation[0].transpose() * K01_local * _total_rotation[0];

  // relation between rotations at node 0 and rotational moments at node 0
  RankTwoTensor K02_local=(1.0/7.0)*K00_local;
  _Kuu02[0] = _total_rotation[0].transpose() * K02_local * _total_rotation[0];

  // relation between rotations at node 0 and rotational moments at node 1
  RankTwoTensor K03_local = (1.0/7.0)*K01_local;
  K03_local(1, 2) = -shear_modulus * A / 6.0;
  K03_local(2, 1) = shear_modulus * A / 6.0;
  _Kut03[0] = _total_rotation[0].transpose() * K03_local * _total_rotation[0];

  // relation between displacements at node 0 and rotational moments at node 1
   RankTwoTensor K04_local=(-8.0/7.0)*K00_local;
  _Kuu04[0] = _total_rotation[0].transpose() * K04_local * _total_rotation[0];

  RankTwoTensor K05_local = (-8.0/7.0)*K01_local;
  K05_local(1, 2) = 2.0* shear_modulus * A / 3.0;
  K05_local(2, 1) = -2.0* shear_modulus * A / 3.0;
  _Kut05[0] = _total_rotation[0].transpose() * K05_local * _total_rotation[0];

  RankTwoTensor K11_local;
  K11_local.zero();
  K11_local(0, 0) = (7.0/3.0)*shear_modulus * Ix / _original_length[0];
  K11_local(0, 1) = (-0.5 )*shear_modulus * Ay;
  K11_local(0, 2) = (-0.5 )*shear_modulus * Az;
  K11_local(1, 0) = (-0.5 )*shear_modulus * Ay;
  K11_local(1, 1) = (1.0/20)*A*shear_modulus*_original_length[0]+youngs_modulus*Iz/_original_length[0]+
                    (1.0/12)*A*shear_modulus*_original_length[0]+(4.0/3.0)*youngs_modulus*Iz/_original_length[0];
  K11_local(2, 0) = (-0.5 )*shear_modulus * Az;
  K11_local(2, 2) = (1.0/20)*A*shear_modulus*_original_length[0]+youngs_modulus*Iy/_original_length[0]+
                    (1.0/12)*A*shear_modulus*_original_length[0]+(4.0/3.0)*youngs_modulus*Iy/_original_length[0];
  _Ktt11[0] = _total_rotation[0].transpose() * K11_local * _total_rotation[0];


 RankTwoTensor K12_local=(1.0/7.0)*K01_local.transpose();
  K12_local(1, 2) = -shear_modulus * A / 6.0;
  K12_local(2, 1) = shear_modulus * A / 6.0;
  _Ktu12[0] = _total_rotation[0].transpose() * K12_local * _total_rotation[0];

 RankTwoTensor K13_local;
  K13_local.zero();
  K13_local(0, 0) = (1.0/3.0)*shear_modulus * Ix / _original_length[0];
  K13_local(0, 1) = (1.0/6.0)*shear_modulus * Ay;
  K13_local(0, 2) = (1.0/6.0)*shear_modulus * Az;
  K13_local(1, 0) = (-1.0/6.0)*shear_modulus * Ay;
  K13_local(1, 1) = (1.0/20)*A*shear_modulus*_original_length[0]-youngs_modulus*Iz/_original_length[0]-
                    (1.0/12)*A*shear_modulus*_original_length[0]+(4.0/3.0)*youngs_modulus*Iz/_original_length[0];
  K13_local(2, 0) = (-1.0/6.0)*shear_modulus * Az;
  K13_local(2, 2) = (1.0/20)*A*shear_modulus*_original_length[0]-youngs_modulus*Iy/_original_length[0]-
                    (1.0/12)*A*shear_modulus*_original_length[0]+(4.0/3.0)*youngs_modulus*Iy/_original_length[0];
  _Ktt13[0] = _total_rotation[0].transpose() * K13_local * _total_rotation[0];

  RankTwoTensor K14_local=(-8.0/7.0)*K01_local.transpose();
  K14_local(1, 2) = 2.0*shear_modulus * A / 3.0;
  K14_local(2, 1) = -2.0*shear_modulus * A / 3.0;
  _Ktu14[0] = _total_rotation[0].transpose() * K14_local * _total_rotation[0];

  RankTwoTensor K15_local;
  K15_local.zero();
  K15_local(0, 0) = (-8.0/3.0)*shear_modulus * Ix / _original_length[0];
  K15_local(0, 1) = (-2.0/3.0)*shear_modulus * Ay;
  K15_local(0, 2) = (-2.0/3.0)*shear_modulus * Az;
  K15_local(1, 0) = (2.0/3.0)*shear_modulus * Ay;
  K15_local(1, 1) = (-1.0/10)*A*shear_modulus*_original_length[0]+
                    (1.0/6.0)*A*shear_modulus*_original_length[0]+(-8.0/3.0)*youngs_modulus*Iz/_original_length[0];
  K15_local(2, 0) = (2.0/3.0)*shear_modulus * Az;
  K15_local(2, 2) = (-1.0/10)*A*shear_modulus*_original_length[0]+
                    (1.0/6.0)*A*shear_modulus*_original_length[0]+(-8.0/3.0)*youngs_modulus*Iy/_original_length[0];
  _Ktt15[0] = _total_rotation[0].transpose() * K15_local * _total_rotation[0];

  RankTwoTensor K22_local=K00_local;
  _Kuu22[0] = _total_rotation[0].transpose() * K22_local * _total_rotation[0];

  RankTwoTensor K23_local=K01_local;
  K23_local(1, 2) = -shear_modulus * A / 2.0;
  K23_local(2, 1) = shear_modulus * A / 2.0;
  _Kut23[0] = _total_rotation[0].transpose() * K23_local * _total_rotation[0];

  RankTwoTensor K24_local=(-8.0/7.0)*K00_local;
  _Kuu24[0] = _total_rotation[0].transpose() * K24_local * _total_rotation[0];

  RankTwoTensor K25_local=(-8.0/7.0)*K01_local;
  K25_local(1, 2) = -2.0*shear_modulus * A / 3.0;
  K25_local(2, 1) = 2.0*shear_modulus * A / 3.0;
  _Kut25[0] = _total_rotation[0].transpose() * K25_local * _total_rotation[0];

  RankTwoTensor K33_local=K11_local;
  K33_local(0, 1) = (1.0/2.0)*shear_modulus * Ay;
  K33_local(0, 2) = (1.0/2.0)*shear_modulus * Az;
  K33_local(1, 0) = (1.0/2.0)*shear_modulus * Ay;
  K33_local(2, 0) = (1.0/2.0)*shear_modulus * Az;
  _Ktt33[0] = _total_rotation[0].transpose() * K33_local * _total_rotation[0];

  RankTwoTensor K34_local=(-8.0/7.0)*K01_local.transpose();
  K34_local(1, 2) = -2.0*shear_modulus * A / 3.0;
  K34_local(2, 1) = 2.0*shear_modulus * A / 3.0;
  _Ktu34[0] = _total_rotation[0].transpose() * K34_local * _total_rotation[0];

  RankTwoTensor K35_local=K15_local;
  K35_local(0, 1) = (2.0/3.0)*shear_modulus * Ay;
  K35_local(0, 2) = (2.0/3.0)*shear_modulus * Az;
  K35_local(1, 0) = -(2.0/3.0)*shear_modulus * Ay;
  K35_local(2, 0) = -(2.0/3.0)*shear_modulus * Az;
  _Ktt35[0] = _total_rotation[0].transpose() * K35_local * _total_rotation[0];

   RankTwoTensor K44_local;
   K44_local.zero();
   K44_local(0, 0) = (16/3.0)*youngs_modulus * A / _original_length[0];
   K44_local(1, 1) = (16/3.0)*shear_modulus * A/ _original_length[0];
   K44_local(2, 2) = (16/3.0)*shear_modulus * A/ _original_length[0];
  _Kuu44[0] = _total_rotation[0].transpose() * K44_local * _total_rotation[0];

  RankTwoTensor K45_local;
  K45_local.zero();
  K45_local(0, 1) = (16/3.0) * Az* youngs_modulus/_original_length[0];
  K45_local(0, 2) = -(16/3.0) * Ay* youngs_modulus/_original_length[0];
  K45_local(1, 0) = -(16/3.0) * Az* shear_modulus/_original_length[0];
  K45_local(2, 0) = (16/3.0) * Ay* shear_modulus/_original_length[0];
  _Kut45[0] = _total_rotation[0].transpose() * K45_local * _total_rotation[0];

  RankTwoTensor K55_local;
   K55_local.zero();
   K55_local(0, 0) = (16/3.0)*shear_modulus * Ix / _original_length[0];
   K55_local(1, 1) = (6.0/5.0)*A*shear_modulus*_original_length[0]-
                    (2.0/3.0)*A*shear_modulus*_original_length[0]+(16.0/3.0)*youngs_modulus*Iz/_original_length[0];
   K55_local(2, 2) = (6.0/5.0)*A*shear_modulus*_original_length[0]-
                    (2.0/3.0)*A*shear_modulus*_original_length[0]+(16.0/3.0)*youngs_modulus*Iy/_original_length[0];
  _Ktt55[0] = _total_rotation[0].transpose() * K55_local * _total_rotation[0];


}

void
ComputeIncrementalCurvedBeamStrain::computeRotation()
{
  _total_rotation[0] = _original_local_config;
}
