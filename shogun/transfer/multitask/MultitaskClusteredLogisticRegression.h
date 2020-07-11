/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */


#ifndef  MULTITASKCLUSTEREDLOGISTICREGRESSION_H_
#define  MULTITASKCLUSTEREDLOGISTICREGRESSION_H_

#include <shogun/lib/config.h>

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

namespace shogun
{
/** @brief class MultitaskClusteredLogisticRegression, a classifier for multitask problems.
 * Supports only task group relations. Based on solver ported from the MALSAR library.
 * Assumes task in group are related with a clustered structure.
 *
 * @see TaskGroup
 */
class MultitaskClusteredLogisticRegression : public MultitaskLogisticRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		MultitaskClusteredLogisticRegression();

		/** constructor
		 *
		 * @param rho1 rho1 regularization coefficient
		 * @param rho2 rho2 regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_group task group
		 * @param num_clusters number of task clusters
		 */
		MultitaskClusteredLogisticRegression(
		     float64_t rho1, float64_t rho2, std::shared_ptr<Features> training_data,
		     std::shared_ptr<BinaryLabels> training_labels, const std::shared_ptr<TaskGroup>& task_group,
		     int32_t num_clusters);

		/** destructor */
		~MultitaskClusteredLogisticRegression() override;

		/** get rho1 regularization coefficient
		 *
		 * @return rho1 value
		 */
		int32_t get_rho1() const;

		/** set rho1
		 * @param rho1 value
		 */
		void set_rho1(float64_t rho1);

		/** get rho1
		 */
		int32_t get_rho2() const;

		/** set rho1
		 * @param rho2 value
		 */
		void set_rho2(float64_t rho2);

		/** get number of clusters
		 *
		 * @return number of clusters
		 */
		int32_t get_num_clusters() const;

		/** set number of clusters
		 * @param num_clusters number of clusters
		 */
		void set_num_clusters(int32_t num_clusters);

		/** get name
		 *
		 * @return name of the object
		 */
		const char* get_name() const override
		{
			return "MultitaskClusteredLogisticRegression";
		}

	protected:

		/** train machine
		 *
		 * @param data features to use for training
		 */
		bool train_machine(std::shared_ptr<Features> data=NULL) override;

		/** train locked implementation
		 *
		 * @param tasks array of tasks indices
		 */
		bool train_locked_implementation(SGVector<index_t>* tasks) override;

	protected:

		/** rho1 */
		float64_t m_rho1;

		/** rho2 */
		float64_t m_rho2;

		/** number of clusters */
		int32_t m_num_clusters;
};
}
#endif
