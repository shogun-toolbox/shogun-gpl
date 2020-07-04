/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskClusteredLogisticRegression.h>

#include <shogun/lib/malsar/malsar_options.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/lib/SGMatrix.h>

#include <utility>

namespace shogun
{

MultitaskClusteredLogisticRegression::MultitaskClusteredLogisticRegression() :
	MultitaskLogisticRegression(), m_rho1(0.0), m_rho2(0.0)
{
}

MultitaskClusteredLogisticRegression::MultitaskClusteredLogisticRegression(
     float64_t rho1, float64_t rho2, const std::shared_ptr<TaskGroup>& task_group, int32_t n_clusters) :
	MultitaskLogisticRegression(0.0, task_group->as<TaskRelation>())
{
	set_rho1(rho1);
	set_rho2(rho2);
	set_num_clusters(n_clusters);
}

int32_t MultitaskClusteredLogisticRegression::get_rho1() const
{
	return m_rho1;
}

int32_t MultitaskClusteredLogisticRegression::get_rho2() const
{
	return m_rho2;
}

void MultitaskClusteredLogisticRegression::set_rho1(float64_t rho1)
{
	m_rho1 = rho1;
}

void MultitaskClusteredLogisticRegression::set_rho2(float64_t rho2)
{
	m_rho2 = rho2;
}

int32_t MultitaskClusteredLogisticRegression::get_num_clusters() const
{
	return m_num_clusters;
}

void MultitaskClusteredLogisticRegression::set_num_clusters(int32_t num_clusters)
{
	m_num_clusters = num_clusters;
}

MultitaskClusteredLogisticRegression::~MultitaskClusteredLogisticRegression()
{
}

bool MultitaskClusteredLogisticRegression::train_locked_implementation(const std::shared_ptr<Features>& data, 
			const std::shared_ptr<Labels>& labs, SGVector<index_t>* tasks)
{
	SGVector<float64_t> y(labs->get_num_labels());
	auto bl = binary_labels(labs);
	const auto features = data->as<DotFeatures>();
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = m_task_relation->as<TaskGroup>()->get_num_tasks();
	options.tasks_indices = tasks;
	options.n_clusters = m_num_clusters;

	io::warn("Clustered LR is unstable with C++11");
	m_tasks_w = SGMatrix<float64_t>(features->as<DotFeatures>()->get_dim_feature_space(), options.n_tasks);
	m_tasks_w.set_const(0);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
	m_tasks_c.set_const(0);

    return true;
}

bool MultitaskClusteredLogisticRegression::train_machine(const std::shared_ptr<Features>& data,
	const std::shared_ptr<Labels>& labs)
{
	const auto features = data->as<DotFeatures>();
	
	require(m_task_relation, "Task relation not set");
	SGVector<float64_t> y(m_labels->get_num_labels());
	auto bl = binary_labels(labs);
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = bl->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = m_task_relation->as<TaskGroup>()->get_num_tasks();
	options.tasks_indices = m_task_relation->as<TaskGroup>()->get_tasks_indices();
	options.n_clusters = m_num_clusters;

	io::warn("Clustered LR is unstable with C++11");
	m_tasks_w = SGMatrix<float64_t>(features->as<DotFeatures>()->get_dim_feature_space(), options.n_tasks);
	m_tasks_w.set_const(0);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
	m_tasks_c.set_const(0);

	SG_FREE(options.tasks_indices);

	return true;
}

}
