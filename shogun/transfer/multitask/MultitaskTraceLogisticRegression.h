/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */


#ifndef  MULTITASKTRACELOGISTICREGRESSION_H_
#define  MULTITASKTRACELOGISTICREGRESSION_H_

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

namespace shogun
{
/** @brief class MultitaskTraceLogisticRegression, a classifier for multitask problems.
 * Supports only task group relations. Based on solver ported from the MALSAR library.
 *
 * @see TaskGroup
 */
class MultitaskTraceLogisticRegression : public MultitaskLogisticRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		MultitaskTraceLogisticRegression();

		/** constructor
		 *
		 * @param rho rho regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 */
		MultitaskTraceLogisticRegression(
		     float64_t rho, const std::shared_ptr<TaskGroup>& task_relation);

		/** destructor */
		~MultitaskTraceLogisticRegression() override;

		/** set rho
		 * @param rho value
		 */
		void set_rho(float64_t rho);

		/** get rho
		 * @return rho value
		 */
		float64_t get_rho() const;

		/** get name
		 *
		 * @return name of the object
		 */
		const char* get_name() const override
		{
			return "MultitaskTraceLogisticRegression";
		}

	private:

		/** init */
		void init();

	protected:

		/** train machine */
		bool train_machine(const std::shared_ptr<DotFeatures>& data, 
			const std::shared_ptr<Labels>& labs) override;

		/** train locked implementation */
		bool train_locked_implementation(const std::shared_ptr<Features>&, 
			const std::shared_ptr<Labels>&, SGVector<index_t>* tasks) override;

	protected:

		/** rho */
		float64_t m_rho;

};
}
#endif //USE_GPL_SHOGUN
#endif

