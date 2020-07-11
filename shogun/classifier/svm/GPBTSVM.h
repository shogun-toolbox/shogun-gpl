/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GPBTSVM_H___
#define _GPBTSVM_H___

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/shogun_libsvm.h>


namespace shogun
{
/** @brief class GPBTSVM */
class GPBTSVM : public SVM
{
	public:
		/** default constructor */
		GPBTSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		GPBTSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab);
		~GPBTSVM() override;

		/** @return object name */
		const char* get_name() const override { return "GPBTSVM"; }

	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		bool train_machine(std::shared_ptr<Features> data=NULL) override;

	protected:
		/** SVM model */
		struct svm_model* model;
};
}
#endif //USE_GPL_SHOGUN
#endif
