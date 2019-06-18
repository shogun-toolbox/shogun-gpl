/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */


#include <shogun/classifier/svm/GPBTSVM.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/external/gpdt.h>
#include <shogun/lib/external/gpdtsolve.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

GPBTSVM::GPBTSVM()
: SVM(), model(NULL)
{
}

GPBTSVM::GPBTSVM(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: SVM(C, k, lab), model(NULL)
{
}

GPBTSVM::~GPBTSVM()
{
	SG_FREE(model);
}

bool GPBTSVM::train_machine(std::shared_ptr<Features> data)
{
	float64_t* solution;                     /* store the solution found       */
	QPproblem prob;                          /* object containing the solvers  */

	ASSERT(kernel)
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_BINARY)
	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			error("Number of training vectors does not match number of labels");
		kernel->init(data, data);
	}

	SGVector<int32_t> lab=m_labels->as<BinaryLabels>()->get_int_labels();
	prob.KER=new sKernel(kernel.get(), lab.vlen);
	prob.y=lab.vector;
	prob.ell=lab.vlen;
	io::info("{} trainlabels", prob.ell);

	//  /*** set options defaults ***/
	prob.delta = epsilon;
	prob.maxmw = kernel->get_cache_size();
	prob.verbosity       = 0;
	prob.preprocess_size = -1;
	prob.projection_projector = -1;
	prob.c_const = get_C1();
	prob.chunk_size = get_qpsize();
	prob.linadd = get_linadd_enabled();

	if (prob.chunk_size < 2)      prob.chunk_size = 2;
	if (prob.q <= 0)              prob.q = prob.chunk_size / 3;
	if (prob.q < 2)               prob.q = 2;
	if (prob.q > prob.chunk_size) prob.q = prob.chunk_size;
	prob.q = prob.q & (~1);
	if (prob.maxmw < 5)
		prob.maxmw = 5;

	/*** set the problem description for final report ***/
	io::info("\nTRAINING PARAMETERS:");
	io::info("\tNumber of training documents: {}", prob.ell);
	io::info("\tq: {}", prob.chunk_size);
	io::info("\tn: {}", prob.q);
	io::info("\tC: {}", prob.c_const);
	io::info("\tkernel type: {}", prob.ker_type);
	io::info("\tcache size: {}Mb", prob.maxmw);
	io::info("\tStopping tolerance: {}", prob.delta);

	//  /*** compute the number of cache rows up to maxmw Mb. ***/
	if (prob.preprocess_size == -1)
		prob.preprocess_size = (int32_t) ( (float64_t)prob.chunk_size * 1.5 );

	if (prob.projection_projector == -1)
	{
		if (prob.chunk_size <= 20) prob.projection_projector = 0;
		else prob.projection_projector = 1;
	}

	/*** compute the problem solution *******************************************/
	solution = SG_MALLOC(float64_t, prob.ell);
	prob.gpdtsolve(solution);
	/****************************************************************************/

	SVM::set_objective(prob.objective_value);

	int32_t num_sv=0;
	int32_t bsv=0;
	int32_t i=0;
	int32_t k=0;

	for (i = 0; i < prob.ell; i++)
	{
		if (solution[i] > prob.DELTAsv)
		{
			num_sv++;
			if (solution[i] > (prob.c_const - prob.DELTAsv)) bsv++;
		}
	}

	create_new_model(num_sv);
	set_bias(prob.bee);

	io::info("SV: {} BSV = {}", num_sv, bsv);

	for (i = 0; i < prob.ell; i++)
	{
		if (solution[i] > prob.DELTAsv)
		{
			set_support_vector(k, i);
			set_alpha(k++, solution[i]*(m_labels->as<BinaryLabels>()->get_label(i)));
		}
	}

	delete prob.KER;
	SG_FREE(solution);

	return true;
}
#endif //USE_GPL_SHOGUN
