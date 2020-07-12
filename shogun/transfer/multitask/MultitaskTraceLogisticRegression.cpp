/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */


#include <shogun/transfer/multitask/MultitaskTraceLogisticRegression.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/malsar/malsar_low_rank.h>
#include <shogun/lib/malsar/malsar_options.h>
#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DotFeatures.h>

#include <utility>

namespace shogun
{

MultitaskTraceLogisticRegression::MultitaskTraceLogisticRegression() :
	MultitaskLogisticRegression(), m_rho(0.0)
{
	init();
}

MultitaskTraceLogisticRegression::MultitaskTraceLogisticRegression(
     float64_t rho, const std::shared_ptr<TaskGroup>& task_group) :
	MultitaskLogisticRegression(0.0, task_group->as<TaskRelation>())
{
	set_rho(rho);
	init();
}

void MultitaskTraceLogisticRegression::init()
{
	SG_ADD(&m_rho,"rho","rho", ParameterProperties::HYPER);
}

void MultitaskTraceLogisticRegression::set_rho(float64_t rho)
{
	m_rho = rho;
}

float64_t MultitaskTraceLogisticRegression::get_rho() const
{
	return m_rho;
}

MultitaskTraceLogisticRegression::~MultitaskTraceLogisticRegression()
{
}

bool MultitaskTraceLogisticRegression::train_locked_implementation(const std::shared_ptr<Features>& data, 
			const std::shared_ptr<Labels>& labs,SGVector<index_t>* tasks)
{
	SGVector<float64_t> y(labs->get_num_labels());
	auto bl = binary_labels(labs);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = m_task_relation->as<TaskGroup>()->get_num_tasks();
	options.tasks_indices = tasks;

	malsar_result_t model = malsar_low_rank(
		data->as<DotFeatures>(), y.vector, m_rho, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
	return true;
}

bool MultitaskTraceLogisticRegression::train_machine(const std::shared_ptr<DotFeatures>& features, 
			const std::shared_ptr<Labels>& labs)
{
	SGVector<float64_t> y(labs->get_num_labels());
	auto bl = binary_labels(labs);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = m_task_relation->as<TaskGroup>()->get_num_tasks();
	options.tasks_indices = m_task_relation->as<TaskGroup>()->get_tasks_indices();

	malsar_result_t model = malsar_low_rank(
		features, y.vector, m_rho, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;

	SG_FREE(options.tasks_indices);

	return true;
}

}

#endif //USE_GPL_SHOGUN
