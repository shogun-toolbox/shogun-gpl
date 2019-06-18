/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */


#ifndef COVERTREESOLVER_H__
#define COVERTREESOLVER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/Distance.h>
#include <shogun/multiclass/KNNSolver.h>

namespace shogun
{

/**
 * Cover tree solver. It uses cover trees to speed up the nearest neighbour computation.
 * For more information, see https://en.wikipedia.org/wiki/Cover_tree
 *
 */
class CoverTreeKNNSolver : public KNNSolver
{
	public:
		/** default constructor */
		CoverTreeKNNSolver() : KNNSolver()
		{ /* nothing to do */ }

		/** deconstructor */
		virtual ~CoverTreeKNNSolver() { /* nothing to do */ }

		/** constructor
		 *
		 * @param k k
		 * @param q m_q
		 * @param num_classes m_num_classes
		 * @param min_label m_min_label
		 * @param train_labels m_train_labels 
		 */
		CoverTreeKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels);

		virtual std::shared_ptr<MulticlassLabels> classify_objects(std::shared_ptr<Distance> d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const;

		virtual SGVector<int32_t> classify_objects_k(std::shared_ptr<Distance> d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const;

		/** @return object name */
		const char* get_name() const { return "CoverTreeKNNSolver"; }

};
}

#endif
