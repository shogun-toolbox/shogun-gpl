/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/classifier/FeatureBlockLogisticRegression.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/slep/slep_options.h>
#include <shogun/lib/slep/slep_solver.h>

#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/IndexBlockTree.h>

#include <utility>

namespace shogun
{

FeatureBlockLogisticRegression::FeatureBlockLogisticRegression() :
	LinearMachine()
{
	init();
	register_parameters();
}

FeatureBlockLogisticRegression::FeatureBlockLogisticRegression(
     float64_t z, std::shared_ptr<IndexBlockRelation> feature_relation) :
	LinearMachine()
{
	init();
	set_feature_relation(std::move(feature_relation));
	set_z(z);
	register_parameters();
}

void FeatureBlockLogisticRegression::init()
{
	m_feature_relation=NULL;
	m_z=0.0;
	m_q=2.0;
	m_termination=0;
	m_regularization=0;
	m_tolerance=1e-3;
	m_max_iter=1000;
}

FeatureBlockLogisticRegression::~FeatureBlockLogisticRegression()
{
}

void FeatureBlockLogisticRegression::register_parameters()
{
	SG_ADD(&m_feature_relation, "feature_relation", "feature relation");
	SG_ADD(&m_z, "z", "regularization coefficient", ParameterProperties::HYPER);
	SG_ADD(&m_q, "q", "q of L1/Lq", ParameterProperties::HYPER);
	SG_ADD(&m_termination, "termination", "termination",
		ParameterProperties::SETTING);
	SG_ADD(&m_regularization, "regularization", "regularization",
		ParameterProperties::SETTING);
	SG_ADD(&m_tolerance, "tolerance", "tolerance",
		ParameterProperties::HYPER);
	SG_ADD(&m_max_iter, "max_iter", "maximum number of iterations",
		ParameterProperties::HYPER);
}

std::shared_ptr<IndexBlockRelation> FeatureBlockLogisticRegression::get_feature_relation() const
{
	return m_feature_relation;
}

void FeatureBlockLogisticRegression::set_feature_relation(std::shared_ptr<IndexBlockRelation> feature_relation)
{
	m_feature_relation = std::move(feature_relation);
}

int32_t FeatureBlockLogisticRegression::get_max_iter() const
{
	return m_max_iter;
}

int32_t FeatureBlockLogisticRegression::get_regularization() const
{
	return m_regularization;
}

int32_t FeatureBlockLogisticRegression::get_termination() const
{
	return m_termination;
}

float64_t FeatureBlockLogisticRegression::get_tolerance() const
{
	return m_tolerance;
}

float64_t FeatureBlockLogisticRegression::get_z() const
{
	return m_z;
}

float64_t FeatureBlockLogisticRegression::get_q() const
{
	return m_q;
}

void FeatureBlockLogisticRegression::set_max_iter(int32_t max_iter)
{
	ASSERT(max_iter>=0)
	m_max_iter = max_iter;
}

void FeatureBlockLogisticRegression::set_regularization(int32_t regularization)
{
	ASSERT(regularization==0 || regularization==1)
	m_regularization = regularization;
}

void FeatureBlockLogisticRegression::set_termination(int32_t termination)
{
	ASSERT(termination>=0 && termination<=4)
	m_termination = termination;
}

void FeatureBlockLogisticRegression::set_tolerance(float64_t tolerance)
{
	ASSERT(tolerance>0.0)
	m_tolerance = tolerance;
}

void FeatureBlockLogisticRegression::set_z(float64_t z)
{
	m_z = z;
}

void FeatureBlockLogisticRegression::set_q(float64_t q)
{
	m_q = q;
}

bool FeatureBlockLogisticRegression::train_machine(const std::shared_ptr<Features>& data,
	 const std::shared_ptr<Labels>& labs)
{
	const auto features = data->as<DotFeatures>();

	int32_t n_vecs = labs->get_num_labels();
	SGVector<float64_t> y(n_vecs);
	for (int32_t i=0; i<n_vecs; i++)
		y[i] = labs->as<BinaryLabels>()->get_label(i);

	slep_options options = slep_options::default_options();
	options.q = m_q;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.loss = LOGISTIC;

	EIndexBlockRelationType relation_type = m_feature_relation->get_relation_type();
	switch (relation_type)
	{
		case GROUP:
		{
			auto feature_group = m_feature_relation->as<IndexBlockGroup>();
			SGVector<index_t> ind = feature_group->get_SLEP_ind();
			options.ind = ind.vector;
			options.n_feature_blocks = ind.vlen-1;
			if (ind[ind.vlen-1] > features->get_dim_feature_space())
				error("Group of features covers more features than available");

			options.gWeight = SG_MALLOC(double, options.n_feature_blocks);
			for (int32_t i=0; i<options.n_feature_blocks; i++)
				options.gWeight[i] = 1.0;
			options.mode = FEATURE_GROUP;
			options.loss = LOGISTIC;
			options.n_nodes = 0;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);

			SG_FREE(options.gWeight);
			int32_t n_feats = features->get_dim_feature_space();
			SGVector<float64_t> new_w(n_feats);
			for (int i=0; i<n_feats; i++)
				new_w[i] = result.w[i];
			set_bias(result.c[0]);

			set_w(new_w);
		}
		break;
		case TREE:
		{
			auto feature_tree = m_feature_relation->as<IndexBlockTree>();

			SGVector<float64_t> ind_t = feature_tree->get_SLEP_ind_t();
			SGVector<float64_t> G;
			if (feature_tree->is_general())
			{
				G = feature_tree->get_SLEP_G();
				options.general = true;
			}
			options.ind_t = ind_t.vector;
			options.G = G.vector;
			options.n_nodes = ind_t.vlen/3;
			options.n_feature_blocks = ind_t.vlen/3;
			options.mode = FEATURE_TREE;
			options.loss = LOGISTIC;

			slep_result_t result = slep_solver(features, y.vector, m_z, options);

			int32_t n_feats = features->get_dim_feature_space();
			SGVector<float64_t> new_w(n_feats);
			for (int i=0; i<n_feats; i++)
				new_w[i] = result.w[i];

			set_bias(result.c[0]);

			set_w(new_w);
		}
		break;
		default:
			error("Not supported feature relation type");
	}

	return true;
}

float64_t FeatureBlockLogisticRegression::apply_one(const std::shared_ptr<DotFeatures>& features, 
	int32_t vec_idx)
{
	SGVector<float64_t> w = get_w();
	return std::exp(-(features->dot(vec_idx, w) + bias));
}

SGVector<float64_t> FeatureBlockLogisticRegression::apply_get_outputs(std::shared_ptr<Features> data)
{
	const auto features = data->as<DotFeatures>();

	int32_t num=features->get_num_vectors();
	SGVector<float64_t> w = get_w();
	ASSERT(num>0)
	ASSERT(w.vlen==features->get_dim_feature_space())

	float64_t* out=SG_MALLOC(float64_t, num);
	features->dense_dot_range(out, 0, num, NULL, w.vector, w.vlen, bias);
	for (int32_t i=0; i<num; i++)
		out[i] = 2.0/(1.0+std::exp(-out[i])) - 1.0;
	return SGVector<float64_t>(out,num);
}

}
#endif //USE_GPL_SHOGUN
