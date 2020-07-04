/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKMACHINE_H_
#define  MULTITASKMACHINE_H_

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/machine/LinearMachine.h>
#include <shogun/transfer/multitask/TaskRelation.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/transfer/multitask/Task.h>

#include <vector>
#include <set>

namespace shogun
{
/** @brief class MultitaskLinearMachine, a base class
 * for linear multitask classifiers
 */
class MultitaskLinearMachine : public LinearMachine
{

	public:
		/** default constructor */
		MultitaskLinearMachine();

		/** constructor
		 *
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 */
		MultitaskLinearMachine(std::shared_ptr<TaskRelation> task_relation);

		/** destructor */
		~MultitaskLinearMachine() override;

		/** get name */
		const char* get_name() const override
		{
			return "MultitaskLinearMachine";
		}

		/** getter for current task
		 * @return current task index
		 */
		int32_t get_current_task() const;

		/** setter for current task
		 * @param task task index
		 */
		void set_current_task(int32_t task);

		/** get w
		 *
		 * @return weight vector
		 */
		SGVector<float64_t> get_w() const override;

		/** set w
		 *
		 * @param src_w new w
		 */
		void set_w(const SGVector<float64_t> src_w) override;

		/** set bias
		 *
		 * @param b new bias
		 */
		void set_bias(float64_t b) override;

		/** get bias
		 *
		 * @return bias
		 */
		float64_t get_bias() const override;

		/** getter for task relation
		 * @return task relation
		 */
		std::shared_ptr<TaskRelation> get_task_relation() const;

		/** setter for task relation
		 * @param task_relation task relation
		 */
		void set_task_relation(std::shared_ptr<TaskRelation> task_relation);

		/** @return whether machine supports locking */
		virtual bool supports_locking() const { return true; }

		/** post lock */
		virtual void post_lock(std::shared_ptr<Labels> labels, std::shared_ptr<Features> features_);

#ifndef SWIG // SWIG should skip this part
		/** train on given indices */
		virtual bool train_locked(const std::shared_ptr<Features>&, 
			const std::shared_ptr<Labels>&, SGVector<index_t> indices);

		/** applies on given indices */
		virtual std::shared_ptr<BinaryLabels> apply_locked_binary(const std::shared_ptr<DotFeatures>&, SGVector<index_t> indices);
#endif // SWIG // SWIG should skip this part

		/** applies to one vector */
<<<<<<< HEAD
		float64_t apply_one(int32_t i) override;
=======
		virtual float64_t apply_one(const std::shared_ptr<DotFeatures>& features, int32_t i);
>>>>>>> refactor linear machine

	protected:

		/** apply get outputs */
		SGVector<float64_t> apply_get_outputs(std::shared_ptr<Features> data=NULL) override;

		/** train machine */
		bool train_machine(std::shared_ptr<Features> data=NULL) override;

		/** train locked implementation */
		virtual bool train_locked_implementation(const std::shared_ptr<Features>&, 
			const std::shared_ptr<Labels>&, SGVector<index_t>* tasks);

		/** subset mapped task indices */
		SGVector<index_t>* get_subset_tasks_indices(const std::shared_ptr<DotFeatures>&);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		/** current task index */
		int32_t m_current_task;

		/** feature tree */
		std::shared_ptr<TaskRelation> m_task_relation;

		/** tasks w's */
		SGMatrix<float64_t> m_tasks_w;

		/** tasks interceptss */
		SGVector<float64_t> m_tasks_c;

		/** vector of sets of indices */
		std::vector< std::set<index_t> > m_tasks_indices;

};
}
#endif //USE_GPL_SHOGUN
#endif
