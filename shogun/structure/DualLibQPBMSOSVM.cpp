/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/structure/DualLibQPBMSOSVM.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/structure/libbmrm.h>
#include <shogun/structure/libppbm.h>
#include <shogun/structure/libp3bm.h>
#include <shogun/structure/libncbm.h>

#include <utility>

using namespace shogun;

DualLibQPBMSOSVM::DualLibQPBMSOSVM()
:LinearStructuredOutputMachine()
{
	init();
}

DualLibQPBMSOSVM::DualLibQPBMSOSVM(
		std::shared_ptr<StructuredModel>	model,
		std::shared_ptr<StructuredLabels>	labs,
		float64_t	_lambda,
		SGVector< float64_t >	W)
 : LinearStructuredOutputMachine(std::move(model), std::move(labs))
{
	init();
	set_lambda(_lambda);

	// Check for initial solution
	if (W.vlen!=0)
	{
		ASSERT(W.size() == m_model->get_dim());
		set_w(W);
	}
}

DualLibQPBMSOSVM::~DualLibQPBMSOSVM()
{
}

void DualLibQPBMSOSVM::init()
{
	SG_ADD(&m_TolRel, "m_TolRel", "Relative tolerance", ParameterProperties::HYPER);
	SG_ADD(&m_TolAbs, "m_TolAbs", "Absolute tolerance", ParameterProperties::HYPER);
	SG_ADD(&m_BufSize, "m_BuffSize", "Size of CP Buffer", ParameterProperties::HYPER);
	SG_ADD(&m_lambda, "m_lambda", "Regularization constant lambda");
	SG_ADD(&m_cleanICP, "m_cleanICP", "Inactive cutting plane removal flag");
	SG_ADD(&m_cleanAfter,
			"m_cleanAfter",
			"Number of inactive iterations after which ICP will be removed");
	SG_ADD(&m_K, "m_K", "Parameter K");
	SG_ADD(&m_Tmax, "m_Tmax", "Parameter Tmax", ParameterProperties::HYPER);
	SG_ADD(&m_cp_models, "m_cp_models", "Number of cutting plane models");

	// TODO(gf712) should be replaced with an extension of Constraint class
	// which has a customisation point with lambdas, rather than write a whole struct
	add_callback_function("w", [&](){
		if (auto dim = m_model->get_dim(); get_w().vlen != dim)
		{
			set_w(SGVector<float64_t>(dim));
			get_w().zero();
			error("Dimension of the initial solution {} must match the model's "
				  "dimension {}.", get_w().vlen, dim);
		}
	});

	set_TolRel(0.001);
	set_TolAbs(0.0);
	set_BufSize(1000);
	set_lambda(0.0);
	set_cleanICP(true);
	set_cleanAfter(10);
	set_K(0.4);
	set_Tmax(100);
	set_cp_models(1);
	set_store_train_info(false);
	set_solver(BMRM);
}

bool DualLibQPBMSOSVM::train_machine(std::shared_ptr<Features> data)
{
	require(m_model, "Model has not been set.");
	if (!m_w)
	{
		set_w(SGVector<float64_t>(m_model->get_dim()));
		get_w().zero();
	}

	if (data)
		set_features(data);

	if (m_verbose||m_store_train_info)
	{
		m_helper = std::make_shared<SOSVMHelper>();
	}

	// Initialize the model for training
	m_model->init_training();
	// call the solver
	switch(m_solver)
	{
		case BMRM:
			m_result=svm_bmrm_solver(this, m_w, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					m_store_train_info);
			break;
		case PPBMRM:
			m_result=svm_ppbm_solver(this, m_w, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					m_verbose);
			break;
		case P3BMRM:
			m_result=svm_p3bm_solver(this, m_w, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, m_K, m_Tmax,
					m_cp_models, m_verbose);
			break;
		case NCBM:
			m_result=svm_ncbm_solver(this, m_w, m_TolRel, m_TolAbs,
					m_lambda, m_BufSize, m_cleanICP, m_cleanAfter, true /* convex */,
					true /* use line search*/, m_verbose);
			break;
		default:
			error("DualLibQPBMSOSVM: m_solver={} is not supported", m_solver);
	}

	if (m_result.exitflag>0)
		return true;
	else
		return false;
}

EMachineType DualLibQPBMSOSVM::get_classifier_type()
{
	return CT_LIBQPSOSVM;
}

#endif //USE_GPL_SHOGUN
