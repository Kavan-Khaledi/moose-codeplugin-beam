//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "StressDivergenceCurvedBeam.h"

// MOOSE includes
#include "Assembly.h"
#include "NonlinearSystem.h"
#include "Material.h"
#include "MooseVariable.h"
#include "SystemBase.h"
#include "RankTwoTensor.h"
#include "MooseMesh.h"

#include "libmesh/quadrature.h"
#include "libmesh/utility.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_type.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/quadrature_gauss.h"

registerMooseObject(MOOSEAPPNAME, StressDivergenceCurvedBeam);

InputParameters
StressDivergenceCurvedBeam::validParams()
{
  InputParameters params = Kernel::validParams();
  params.addClassDescription("Quasi-static and dynamic stress divergence kernel for Beam element");
  params.addRequiredParam<unsigned int>(
      "component",
      "An integer corresponding to the direction "
      "the variable this kernel acts in. (0 for disp_x, "
      "1 for disp_y, 2 for disp_z, 3 for rot_x, 4 for rot_y and 5 for rot_z)");
  params.addRequiredCoupledVar(
      "displacements",
      "The displacements appropriate for the simulation geometry and coordinate system");
  params.addRequiredCoupledVar(
      "rotations", "The rotations appropriate for the simulation geometry and coordinate system");
  params.addParam<MaterialPropertyName>(
      "zeta",
      0.0,
      "Name of material property or a constant real number defining the zeta parameter for the "
      "Rayleigh damping.");
  params.addRangeCheckedParam<Real>(
      "alpha", 0.0, "alpha >= -0.3333 & alpha <= 0.0", "alpha parameter for HHT time integration");

  params.set<bool>("use_displaced_mesh") = true;
  return params;
}

StressDivergenceCurvedBeam::StressDivergenceCurvedBeam(const InputParameters & parameters)
  : Kernel(parameters),
    _component(getParam<unsigned int>("component")),
    _ndisp(coupledComponents("displacements")),
    _disp_var(_ndisp),
    _nrot(coupledComponents("rotations")),
    _rot_var(_nrot),
    _force(getMaterialPropertyByName<RealVectorValue>("forces")),
    _moment(getMaterialPropertyByName<RealVectorValue>("moments")),
    _Kuu00(getMaterialPropertyByName<RankTwoTensor>("Jacobian_00")),
    _Kut01(getMaterialPropertyByName<RankTwoTensor>("Jacobian_01")),
    _Kuu02(getMaterialPropertyByName<RankTwoTensor>("Jacobian_02")),
    _Kut03(getMaterialPropertyByName<RankTwoTensor>("Jacobian_03")),
    _Kuu04(getMaterialPropertyByName<RankTwoTensor>("Jacobian_04")),
    _Kut05(getMaterialPropertyByName<RankTwoTensor>("Jacobian_05")),
    _Ktt11(getMaterialPropertyByName<RankTwoTensor>("Jacobian_11")),
    _Ktu12(getMaterialPropertyByName<RankTwoTensor>("Jacobian_12")),
    _Ktt13(getMaterialPropertyByName<RankTwoTensor>("Jacobian_13")),
    _Ktu14(getMaterialPropertyByName<RankTwoTensor>("Jacobian_14")),
    _Ktt15(getMaterialPropertyByName<RankTwoTensor>("Jacobian_15")),
    _Kuu22(getMaterialPropertyByName<RankTwoTensor>("Jacobian_22")),
    _Kut23(getMaterialPropertyByName<RankTwoTensor>("Jacobian_23")),
    _Kuu24(getMaterialPropertyByName<RankTwoTensor>("Jacobian_24")),
    _Kut25(getMaterialPropertyByName<RankTwoTensor>("Jacobian_25")),
    _Ktt33(getMaterialPropertyByName<RankTwoTensor>("Jacobian_33")),
    _Ktu34(getMaterialPropertyByName<RankTwoTensor>("Jacobian_34")),
    _Ktt35(getMaterialPropertyByName<RankTwoTensor>("Jacobian_35")),
    _Kuu44(getMaterialPropertyByName<RankTwoTensor>("Jacobian_44")),
    _Kut45(getMaterialPropertyByName<RankTwoTensor>("Jacobian_45")),
    _Ktt55(getMaterialPropertyByName<RankTwoTensor>("Jacobian_55")),
    _original_length(getMaterialPropertyByName<Real>("original_length")),
    _total_rotation(getMaterialPropertyByName<RankTwoTensor>("total_rotation")),
    _zeta(getMaterialProperty<Real>("zeta")),
    _alpha(getParam<Real>("alpha")),
    _isDamped(getParam<MaterialPropertyName>("zeta") != "0.0" || std::abs(_alpha) > 0.0),
    _force_old(_isDamped ? &getMaterialPropertyOld<RealVectorValue>("forces") : nullptr),
    _moment_old(_isDamped ? &getMaterialPropertyOld<RealVectorValue>("moments") : nullptr),
    _total_rotation_old(_isDamped ? &getMaterialPropertyOld<RankTwoTensor>("total_rotation")
                                  : nullptr),
    _force_older(std::abs(_alpha) > 0.0 ? &getMaterialPropertyOlder<RealVectorValue>("forces")
                                        : nullptr),
    _moment_older(std::abs(_alpha) > 0.0 ? &getMaterialPropertyOlder<RealVectorValue>("moments")
                                         : nullptr),
    _total_rotation_older(std::abs(_alpha) > 0.0
                              ? &getMaterialPropertyOlder<RankTwoTensor>("total_rotation")
                              : nullptr),
    _global_force_res(0),
    _global_moment_res(0),
    _force_local_t(0),
    _moment_local_t(0),
    _local_force_res(0),
    _local_moment_res(0),
    _nonlinear_sys(_fe_problem.getNonlinearSystemBase(/*nl_sys_num=*/0))
{
  if (_ndisp != _nrot)
    mooseError("StressDivergenceCurvedBeam: The number of displacement and rotation variables "
               "should be same.");

  for (unsigned int i = 0; i < _ndisp; ++i)
    _disp_var[i] = coupled("displacements", i);

  for (unsigned int i = 0; i < _nrot; ++i)
    _rot_var[i] = coupled("rotations", i);
}

void
StressDivergenceCurvedBeam::computeResidual()
{
  prepareVectorTag(_assembly, _var.number());

  mooseAssert(_local_re.size() == 3, "this element works only for 3-noded beam");
  _global_force_res.resize(_test.size());
  _global_moment_res.resize(_test.size());

  computeGlobalResidual(&_force, &_moment, &_total_rotation, _global_force_res, _global_moment_res);

  for (_i = 0; _i < _test.size(); ++_i)
  {
    if (_component < 3)
      _local_re(_i) = _global_force_res[_i](_component);
    else
      _local_re(_i) = _global_moment_res[_i](_component - 3);
  }

  accumulateTaggedLocalResidual();

  if (_has_save_in)
  {
    Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);
    for (_i = 0; _i < _save_in.size(); ++_i)
      _save_in[_i]->sys().solution().add_vector(_local_re, _save_in[_i]->dofIndices());
  }
}

void
StressDivergenceCurvedBeam::computeJacobian()
{
  prepareMatrixTag(_assembly, _var.number(), _var.number());

  if (_component < 3)
  {
    _local_ke(0, 0) = _Kuu00[0](_component, _component);
    _local_ke(0, 1) = _Kuu02[0](_component, _component);
    _local_ke(0, 2) = _Kuu04[0](_component, _component);
    _local_ke(1, 0) = _Kuu02[0](_component, _component);
    _local_ke(1, 1) = _Kuu22[0](_component, _component);
    _local_ke(1, 2) = _Kuu24[0](_component, _component);
    _local_ke(2, 0) = _Kuu04[0](_component, _component);
    _local_ke(2, 1) = _Kuu24[0](_component, _component);
    _local_ke(2, 2) = _Kuu44[0](_component, _component);
  }
  else
  {
    _local_ke(0, 0) = _Ktt11[0](_component - 3, _component - 3);
    _local_ke(0, 1) = _Ktt13[0](_component - 3, _component - 3);
    _local_ke(0, 2) = _Ktt15[0](_component - 3, _component - 3);
    _local_ke(1, 0) = _Ktt13[0](_component - 3, _component - 3);
    _local_ke(1, 1) = _Ktt33[0](_component - 3, _component - 3);
    _local_ke(1, 2) = _Ktt35[0](_component - 3, _component - 3);
    _local_ke(2, 0) = _Ktt15[0](_component - 3, _component - 3);
    _local_ke(2, 1) = _Ktt35[0](_component - 3, _component - 3);
    _local_ke(2, 2) = _Ktt55[0](_component - 3, _component - 3);
  }

  accumulateTaggedLocalMatrix();

  if (_has_diag_save_in)
  {
    unsigned int rows = _local_ke.m();
    DenseVector<Number> diag(rows);
    for (unsigned int i = 0; i < rows; ++i)
      diag(i) = _local_ke(i, i);

    Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);
    for (unsigned int i = 0; i < _diag_save_in.size(); ++i)
      _diag_save_in[i]->sys().solution().add_vector(diag, _diag_save_in[i]->dofIndices());
  }
}

void
StressDivergenceCurvedBeam::computeOffDiagJacobian(const unsigned int jvar_num)
{
  if (jvar_num == _var.number())
    computeJacobian();
  else
  {
    unsigned int coupled_component = 0;
    bool disp_coupled = false;
    bool rot_coupled = false;

    for (unsigned int i = 0; i < _ndisp; ++i)
    {
      if (jvar_num == _disp_var[i])
      {
        coupled_component = i;
        disp_coupled = true;
        break;
      }
    }

    for (unsigned int i = 0; i < _nrot; ++i)
    {
      if (jvar_num == _rot_var[i])
      {
        coupled_component = i + 3;
        rot_coupled = true;
        break;
      }
    }

    prepareMatrixTag(_assembly, _var.number(), jvar_num);

    if (disp_coupled || rot_coupled)
    {

      if (_component < 3 && coupled_component < 3)
      {
        _local_ke(0, 0) += _Kuu00[0](_component, coupled_component);
        _local_ke(0, 1) += _Kuu02[0](_component, coupled_component);
        _local_ke(0, 2) += _Kuu04[0](_component, coupled_component);
        _local_ke(1, 0) += _Kuu02[0](coupled_component, _component);
        _local_ke(1, 1) += _Kuu22[0](_component, coupled_component);
        _local_ke(1, 2) += _Kuu24[0](_component, coupled_component);
        _local_ke(2, 0) += _Kuu04[0](coupled_component, _component);
        _local_ke(2, 1) += _Kuu24[0](coupled_component, _component);
        _local_ke(2, 2) += _Kuu44[0](_component, coupled_component);
      }
      else if (_component < 3 && coupled_component > 2)
      {
        _local_ke(0, 0) += _Kut01[0](_component, coupled_component - 3);
        _local_ke(0, 1) += _Kut03[0](_component, coupled_component - 3);
        _local_ke(0, 2) += _Kut05[0](_component, coupled_component - 3);
        _local_ke(1, 0) += _Ktu12[0](coupled_component - 3, _component);
        _local_ke(1, 1) += _Kut23[0](_component, coupled_component - 3);
        _local_ke(1, 2) += _Kut25[0](_component, coupled_component - 3);
        _local_ke(2, 0) += _Ktu14[0](coupled_component - 3, _component);
        _local_ke(2, 1) += _Ktu34[0](coupled_component - 3, _component);
        _local_ke(2, 2) += _Kut45[0](_component, coupled_component - 3);
      }
      else if (_component > 2 && coupled_component < 3)
      {
        _local_ke(0, 0) += _Kut01[0](coupled_component, _component - 3);
        _local_ke(0, 1) += _Ktu12[0](_component - 3, coupled_component);
        _local_ke(0, 2) += _Ktu14[0](_component - 3, coupled_component);
        _local_ke(1, 0) += _Kut03[0](coupled_component, _component - 3);
        _local_ke(1, 1) += _Kut23[0](coupled_component, _component - 3);
        _local_ke(1, 2) += _Ktu34[0](_component - 3, coupled_component);
        _local_ke(2, 0) += _Kut05[0](coupled_component, _component - 3);
        _local_ke(2, 1) += _Kut25[0](coupled_component, _component - 3);
        _local_ke(2, 2) += _Kut45[0](coupled_component, _component - 3);
      }
      else
      {
        _local_ke(0, 0) += _Ktt11[0](_component - 3, coupled_component - 3);
        _local_ke(0, 1) += _Ktt13[0](_component - 3, coupled_component - 3);
        _local_ke(0, 2) += _Ktt15[0](_component - 3, coupled_component - 3);
        _local_ke(1, 0) += _Ktt13[0](coupled_component - 3, _component - 3);
        _local_ke(1, 1) += _Ktt33[0](_component - 3, coupled_component - 3);
        _local_ke(1, 2) += _Ktt35[0](_component - 3, coupled_component - 3);
        _local_ke(2, 0) += _Ktt15[0](coupled_component - 3, _component - 3);
        _local_ke(2, 1) += _Ktt35[0](coupled_component - 3, _component - 3);
        _local_ke(2, 2) += _Ktt55[0](_component - 3, coupled_component - 3);
      }
    }

    accumulateTaggedLocalMatrix();
  }
}

void
StressDivergenceCurvedBeam::computeDynamicTerms(std::vector<RealVectorValue> & global_force_res,
                                                std::vector<RealVectorValue> & global_moment_res)
{
  mooseAssert(_zeta[0] >= 0.0,
              "StressDivergenceCurvedBeam: Zeta parameter should be non-negative.");
  std::vector<RealVectorValue> global_force_res_old(_test.size());
  std::vector<RealVectorValue> global_moment_res_old(_test.size());
  computeGlobalResidual(
      _force_old, _moment_old, _total_rotation_old, global_force_res_old, global_moment_res_old);

  // For HHT calculation, the global force and moment residual from t_older is required
  std::vector<RealVectorValue> global_force_res_older(_test.size());
  std::vector<RealVectorValue> global_moment_res_older(_test.size());

  if (std::abs(_alpha) > 0.0)
    computeGlobalResidual(_force_older,
                          _moment_older,
                          _total_rotation_older,
                          global_force_res_older,
                          global_moment_res_older);

  // Update the global_force_res and global_moment_res to include HHT and Rayleigh damping
  // contributions
  for (_i = 0; _i < _test.size(); ++_i)
  {
    global_force_res[_i] =
        global_force_res[_i] * (1.0 + _alpha + (1.0 + _alpha) * _zeta[0] / _dt) -
        global_force_res_old[_i] * (_alpha + (1.0 + 2.0 * _alpha) * _zeta[0] / _dt) +
        global_force_res_older[_i] * (_alpha * _zeta[0] / _dt);
    global_moment_res[_i] =
        global_moment_res[_i] * (1.0 + _alpha + (1.0 + _alpha) * _zeta[0] / _dt) -
        global_moment_res_old[_i] * (_alpha + (1.0 + 2.0 * _alpha) * _zeta[0] / _dt) +
        global_moment_res_older[_i] * (_alpha * _zeta[0] / _dt);
  }
}

void
StressDivergenceCurvedBeam::computeGlobalResidual(
    const MaterialProperty<RealVectorValue> * force,
    const MaterialProperty<RealVectorValue> * moment,
    const MaterialProperty<RankTwoTensor> * total_rotation,
    std::vector<RealVectorValue> & global_force_res,
    std::vector<RealVectorValue> & global_moment_res)
{
  RealVectorValue a;
  _force_local_t.resize(_qrule->n_points());
  _moment_local_t.resize(_qrule->n_points());
  _local_force_res.resize(_test.size());
  _local_moment_res.resize(_test.size());
  _q_weights = _qrule->get_weights();
  FEType fe_type(Utility::string_to_enum<Order>("SECOND"),
                 Utility::string_to_enum<FEFamily>("LAGRANGE"));
  auto & fe = _fe_problem.assembly(_tid, _nonlinear_sys.number()).getFE(fe_type, 1);
  //{{-1.27459,-0.5,0.27459},{-0.27459,0.5,1.27459},{1.54918,0,-1.54918}};//
  _dphidxi_map = fe->get_fe_map().get_dphidxi_map();
  // {{0.687289,0,-0.0873},{-0.0873,0,0.687289},{0.4,1.0,0.4}};
  _phi_map = fe->get_fe_map().get_phi_map();
  // convert forces/moments from global coordinate system to current beam local configuration
  for (_qp = 0; _qp < _qrule->n_points(); ++_qp)
  {
    _force_local_t[_qp] = (*total_rotation)[0] * (*force)[_qp];
    _moment_local_t[_qp] = (*total_rotation)[0] * (*moment)[_qp];
  }

  // residual for displacement variables
  for (_i = 0; _i < _test.size(); ++_i)
  {
    _local_force_res[_i] = a;
    for (unsigned int component = 0; component < 3; ++component)
    {
      for (_qp = 0; _qp < _qrule->n_points(); ++_qp)
        _local_force_res[_i](component) +=
            _force_local_t[_qp](component) * _q_weights[_qp] * _dphidxi_map[_i][_qp];
    }
  }

  // residual for rotation variables
  for (_i = 0; _i < _test.size(); ++_i)
  {
    _local_moment_res[_i] = a;
    for (unsigned int component = 3; component < 6; ++component)
    {
      for (_qp = 0; _qp < _qrule->n_points(); ++_qp)
      {
        if (component == 3)
          _local_moment_res[_i](component - 3) +=
              _moment_local_t[_qp](0) * _q_weights[_qp] * _dphidxi_map[_i][_qp];
        else if (component == 4)
          _local_moment_res[_i](component - 3) +=
              _moment_local_t[_qp](1) * _q_weights[_qp] * _dphidxi_map[_i][_qp] +
              _force_local_t[_qp](2) * 0.5 * _original_length[0] * _q_weights[_qp] *
                  _phi_map[_i][_qp];
        else
          _local_moment_res[_i](component - 3) +=
              _moment_local_t[_qp](2) * _q_weights[_qp] * _dphidxi_map[_i][_qp] -
              _force_local_t[_qp](1) * 0.5 * _original_length[0] * _q_weights[_qp] *
                  _phi_map[_i][_qp];
        ;
      }
    }
  }

  // convert residual for each variable from current beam local configuration to global
  // configuration
  for (_i = 0; _i < _test.size(); ++_i)
  {
    global_force_res[_i] = (*total_rotation)[0].transpose() * _local_force_res[_i];
    global_moment_res[_i] = (*total_rotation)[0].transpose() * _local_moment_res[_i];
  }
}
