/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */


#include <shogun/transfer/multitask/MultitaskL12LogisticRegression.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/malsar/malsar_joint_feature_learning.h>
#include <shogun/lib/malsar/malsar_options.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{

class MultitaskL12LogisticRegression::Self
{
public:

	/** rho1, regularization coefficient of L1/L2 term */
	float64_t m_rho1;

	/** rho2, regularization coefficient of L2 term */
	float64_t m_rho2;
};

MultitaskL12LogisticRegression::MultitaskL12LogisticRegression() :
	MultitaskLogisticRegression(), self()
{
	set_rho1(0.0);
	set_rho2(0.0);
	init();
}

MultitaskL12LogisticRegression::MultitaskL12LogisticRegression(
     float64_t rho1, float64_t rho2, std::shared_ptr<Features> train_features,
     std::shared_ptr<BinaryLabels> train_labels, std::shared_ptr<TaskGroup> task_group) :
	MultitaskLogisticRegression(0.0,train_features,train_labels,task_group->as<TaskRelation>())
{
	set_rho1(rho1);
	set_rho2(rho2);
	init();
}

void MultitaskL12LogisticRegression::init()
{
	SG_ADD(&self->m_rho1,"rho1","rho L1/L2 regularization parameter", ParameterProperties::HYPER);
	SG_ADD(&self->m_rho2,"rho2","rho L2 regularization parameter", ParameterProperties::HYPER);
}

void MultitaskL12LogisticRegression::set_rho1(float64_t rho1)
{
	self->m_rho1 = rho1;
}

void MultitaskL12LogisticRegression::set_rho2(float64_t rho2)
{
	self->m_rho2 = rho2;
}

float64_t MultitaskL12LogisticRegression::get_rho1() const
{
	return self->m_rho1;
}

float64_t MultitaskL12LogisticRegression::get_rho2() const
{
	return self->m_rho2;
}

MultitaskL12LogisticRegression::~MultitaskL12LogisticRegression()
{
}

bool MultitaskL12LogisticRegression::train_locked_implementation(SGVector<index_t>* tasks)
{
	SGVector<float64_t> y(m_labels->get_num_labels());
	auto bl = binary_labels(m_labels);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = m_task_relation->as<TaskGroup>()->get_num_tasks();
	options.tasks_indices = tasks;
	malsar_result_t model = malsar_joint_feature_learning(
		features, y.vector, self->m_rho1, self->m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;

	return true;
}

bool MultitaskL12LogisticRegression::train_machine(std::shared_ptr<Features> data)
{
	if (data)
		set_features(data->as<DotFeatures>());

	ASSERT(features)
	ASSERT(m_labels)
	ASSERT(m_task_relation)

	SGVector<float64_t> y(m_labels->get_num_labels());
	auto bl = binary_labels(m_labels);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = m_task_relation->as<TaskGroup>()->get_num_tasks();
	options.tasks_indices = m_task_relation->as<TaskGroup>()->get_tasks_indices();

	malsar_result_t model = malsar_joint_feature_learning(
		features, y.vector, self->m_rho1, self->m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;

	SG_FREE(options.tasks_indices);

	return true;
}

}

#endif //USE_GPL_SHOGUN
