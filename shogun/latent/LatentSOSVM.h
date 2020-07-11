/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTSOSVM_H__
#define __LATENTSOSVM_H__

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN

#include <shogun/machine/LinearLatentMachine.h>
#include <shogun/machine/LinearStructuredOutputMachine.h>

namespace shogun
{
	/** @brief class Latent Structured Output SVM,
	 * an structured output based machine for classification
	 * problems with latent variables.
	 */
	class LatentSOSVM: public LinearLatentMachine
	{
		public:
			/** default ctor*/
			LatentSOSVM();

			/**
			 *
			 * @param model
			 * @param so_solver
			 * @param C
			 */
			LatentSOSVM(std::shared_ptr<LatentModel> model, std::shared_ptr<LinearStructuredOutputMachine> so_solver, float64_t C);

			~LatentSOSVM() override;

			/** apply linear machine to data
			 *
			 * @return classified labels
			 */
			std::shared_ptr<LatentLabels> apply_latent() override;

			/** set SO solver that is going to be used
			 *
			 * @param so SO machine
			 */
			void set_so_solver(std::shared_ptr<LinearStructuredOutputMachine> so);

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			const char* get_name() const override { return "LatentSOSVM"; }

		protected:
			/** do inner loop with given cooling epsilon
			 *
			 * @param cooling_eps cooling epsilon
			 */
			float64_t do_inner_loop(float64_t cooling_eps) override;

		private:
			void register_parameters();

		private:
			/** Linear Structured Solver */
			std::shared_ptr<LinearStructuredOutputMachine> m_so_solver;
	};
}
#endif //USE_GPL_SHOGUN
#endif /* __LATENTSOSVM_H__ */

