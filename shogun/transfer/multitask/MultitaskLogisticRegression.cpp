/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */


#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/slep/slep_options.h>
#include <shogun/lib/slep/slep_solver.h>
#include <utility>
#include <vector>

namespace shogun
{

MultitaskLogisticRegression::MultitaskLogisticRegression() :
	MultitaskLinearMachine()
{
	initialize_parameters();
	register_parameters();
}

MultitaskLogisticRegression::MultitaskLogisticRegression(
     float64_t z, std::shared_ptr<Features> train_features,
     const std::shared_ptr<BinaryLabels>& train_labels, std::shared_ptr<TaskRelation> task_relation) :
	MultitaskLinearMachine(std::move(train_features),train_labels,std::move(task_relation))
{
	initialize_parameters();
	register_parameters();
	set_z(z);
}

MultitaskLogisticRegression::~MultitaskLogisticRegression()
{
}

void MultitaskLogisticRegression::register_parameters()
{
	SG_ADD(&m_z, "z", "regularization coefficient", ParameterProperties::HYPER);
	SG_ADD(&m_q, "q", "q of L1/Lq", ParameterProperties::HYPER);
	SG_ADD(&m_termination, "termination", "termination");
	SG_ADD(&m_regularization, "regularization", "regularization");
	SG_ADD(&m_tolerance, "tolerance", "tolerance");
	SG_ADD(&m_max_iter, "max_iter", "maximum number of iterations");
}

void MultitaskLogisticRegression::initialize_parameters()
{
	set_z(0.0);
	set_q(2.0);
	set_termination(0);
	set_regularization(0);
	set_tolerance(1e-3);
	set_max_iter(1000);
}

bool MultitaskLogisticRegression::train_machine(std::shared_ptr<Features> data)
{
	if (data)
		set_features(data->as<DotFeatures>());

	ASSERT(features)
	ASSERT(m_labels)

	SGVector<float64_t> y(m_labels->get_num_labels());
	auto bl = binary_labels(m_labels);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	slep_options options = slep_options::default_options();
	options.n_tasks = m_task_relation->get_num_tasks();
	options.tasks_indices = m_task_relation->get_tasks_indices();
	options.q = m_q;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;

	ETaskRelationType relation_type = m_task_relation->get_relation_type();
	switch (relation_type)
	{
		case TASK_GROUP:
		{
			//TaskGroup* task_group = (TaskGroup*)m_task_relation;
			options.mode = MULTITASK_GROUP;
			options.loss = LOGISTIC;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		case TASK_TREE:
		{
			auto task_tree = m_task_relation->as<TaskTree>();
			SGVector<float64_t> ind_t = task_tree->get_SLEP_ind_t();
			options.ind_t = ind_t.vector;
			options.n_nodes = ind_t.vlen / 3;
			options.mode = MULTITASK_TREE;
			options.loss = LOGISTIC;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		default:
			error("Not supported task relation type");
	}
	SG_FREE(options.tasks_indices);

	return true;
}

bool MultitaskLogisticRegression::train_locked_implementation(SGVector<index_t>* tasks)
{
	ASSERT(features)
	ASSERT(m_labels)

	SGVector<float64_t> y(m_labels->get_num_labels());
	auto bl = binary_labels(m_labels);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	slep_options options = slep_options::default_options();
	options.n_tasks = m_task_relation->get_num_tasks();
	options.tasks_indices = tasks;
	options.q = m_q;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;

	ETaskRelationType relation_type = m_task_relation->get_relation_type();
	switch (relation_type)
	{
		case TASK_GROUP:
		{
			//TaskGroup* task_group = (TaskGroup*)m_task_relation;
			options.mode = MULTITASK_GROUP;
			options.loss = LOGISTIC;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		case TASK_TREE:
		{
			auto task_tree = m_task_relation->as<TaskTree>();
			SGVector<float64_t> ind_t = task_tree->get_SLEP_ind_t();
			options.ind_t = ind_t.vector;
			options.n_nodes = ind_t.vlen / 3;
			options.mode = MULTITASK_TREE;
			options.loss = LOGISTIC;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		default:
			error("Not supported task relation type");
	}
	return true;
}

float64_t MultitaskLogisticRegression::apply_one(int32_t i)
{
	float64_t dot = features->dot(i,m_tasks_w.get_column(m_current_task));
	//float64_t ep = Math::exp(-(dot + m_tasks_c[m_current_task]));
	//return 2.0/(1.0+ep) - 1.0;
	return dot + m_tasks_c[m_current_task];
}

int32_t MultitaskLogisticRegression::get_max_iter() const
{
	return m_max_iter;
}
int32_t MultitaskLogisticRegression::get_regularization() const
{
	return m_regularization;
}
int32_t MultitaskLogisticRegression::get_termination() const
{
	return m_termination;
}
float64_t MultitaskLogisticRegression::get_tolerance() const
{
	return m_tolerance;
}
float64_t MultitaskLogisticRegression::get_z() const
{
	return m_z;
}
float64_t MultitaskLogisticRegression::get_q() const
{
	return m_q;
}

void MultitaskLogisticRegression::set_max_iter(int32_t max_iter)
{
	ASSERT(max_iter>=0)
	m_max_iter = max_iter;
}
void MultitaskLogisticRegression::set_regularization(int32_t regularization)
{
	ASSERT(regularization==0 || regularization==1)
	m_regularization = regularization;
}
void MultitaskLogisticRegression::set_termination(int32_t termination)
{
	ASSERT(termination>=0 && termination<=4)
	m_termination = termination;
}
void MultitaskLogisticRegression::set_tolerance(float64_t tolerance)
{
	ASSERT(tolerance>0.0)
	m_tolerance = tolerance;
}
void MultitaskLogisticRegression::set_z(float64_t z)
{
	m_z = z;
}
void MultitaskLogisticRegression::set_q(float64_t q)
{
	m_q = q;
}

}

#endif //USE_GPL_SHOGUN
