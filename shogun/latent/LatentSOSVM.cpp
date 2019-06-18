/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/latent/LatentSOSVM.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/structure/DualLibQPBMSOSVM.h>

using namespace shogun;

LatentSOSVM::LatentSOSVM()
	: LinearLatentMachine()
{
	register_parameters();
	m_so_solver=NULL;
}

LatentSOSVM::LatentSOSVM(std::shared_ptr<LatentModel> model, std::shared_ptr<LinearStructuredOutputMachine> so_solver, float64_t C)
	: LinearLatentMachine(model, C)
{
	register_parameters();
	set_so_solver(so_solver);
}

LatentSOSVM::~LatentSOSVM()
{
}

std::shared_ptr<LatentLabels> LatentSOSVM::apply_latent()
{
	return NULL;
}

void LatentSOSVM::set_so_solver(std::shared_ptr<LinearStructuredOutputMachine> so)
{
	m_so_solver = so;
}

float64_t LatentSOSVM::do_inner_loop(float64_t cooling_eps)
{
	float64_t lambda = 1/m_C;
	auto so = std::shared_ptr<DualLibQPBMSOSVM>();
	so->set_lambda(lambda);
	so->train();

	/* copy the resulting w */
	set_w(so->get_w().clone());

	/* get the primal objective value */
	float64_t po = so->get_result().Fp;

	return po;
}

void LatentSOSVM::register_parameters()
{
	m_parameters->add((SGObject**)&m_so_solver, "so_solver", "Structured Output Solver.");
}

#endif //USE_GPL_SHOGUN
