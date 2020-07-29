/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */


#include <shogun/multiclass/MulticlassTreeGuidedLogisticRegression.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/slep/slep_mc_tree_lr.h>

#include <utility>

using namespace shogun;

MulticlassTreeGuidedLogisticRegression::MulticlassTreeGuidedLogisticRegression() :
	LinearMulticlassMachine()
{
	init_defaults();
}

MulticlassTreeGuidedLogisticRegression::MulticlassTreeGuidedLogisticRegression(float64_t z, std::shared_ptr<IndexBlockTree> tree) :
	LinearMulticlassMachine(std::make_shared<MulticlassOneVsRestStrategy>(), nullptr)
{
	init_defaults();
	set_z(z);
	set_index_tree(std::move(tree));
}

void MulticlassTreeGuidedLogisticRegression::init_defaults()
{
	m_index_tree = NULL;
	set_z(0.1);
	set_epsilon(1e-2);
	set_max_iter(10000);
}

void MulticlassTreeGuidedLogisticRegression::register_parameters()
{
	SG_ADD(&m_z, "m_z", "regularization constant", ParameterProperties::HYPER);
	SG_ADD(&m_epsilon, "m_epsilon", "tolerance epsilon");
	SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations");
}

MulticlassTreeGuidedLogisticRegression::~MulticlassTreeGuidedLogisticRegression()
{
}

bool MulticlassTreeGuidedLogisticRegression::train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
{
	require(m_multiclass_strategy, "Multiclass strategy not set.");
	require(m_index_tree, "Index tree not set.");
	auto features = data->as<DotFeatures>();
	int32_t n_classes = labs->as<MulticlassLabels>()->get_num_classes();
	int32_t n_feats = features->get_dim_feature_space();

	slep_options options = slep_options::default_options();
	if (!m_machines.empty())
	{
		SGMatrix<float64_t> all_w_old(n_feats, n_classes);
		SGVector<float64_t> all_c_old(n_classes);
		for (int32_t i=0; i<n_classes; i++)
		{
			auto machine = m_machines.at(i)->as<LinearMachine>();
			SGVector<float64_t> w = machine->get_w();
			for (int32_t j=0; j<n_feats; j++)
				all_w_old(j,i) = w[j];
			all_c_old[i] = machine->get_bias();
		}
		options.last_result = new slep_result_t(all_w_old,all_c_old);
		m_machines.clear();
	}
	if (m_index_tree->is_general())
	{
		SGVector<float64_t> G = m_index_tree->get_SLEP_G();
		options.G = G.vector;
	}
	SGVector<float64_t> ind_t = m_index_tree->get_SLEP_ind_t();
	options.ind_t = ind_t.vector;
	options.n_nodes = ind_t.size()/3;
	options.tolerance = m_epsilon;
	options.max_iter = m_max_iter;
	slep_result_t result = slep_mc_tree_lr(features,labs->as<MulticlassLabels>(),m_z,options);

	SGMatrix<float64_t> all_w = result.w;
	SGVector<float64_t> all_c = result.c;
	for (int32_t i=0; i<n_classes; i++)
	{
		SGVector<float64_t> w(n_feats);
		for (int32_t j=0; j<n_feats; j++)
			w[j] = all_w(j,i);
		float64_t c = all_c[i];
		auto machine = std::make_shared<LinearMachine>();
		machine->set_w(w);
		machine->set_bias(c);
		m_machines.push_back(machine);
	}
	return true;
}
