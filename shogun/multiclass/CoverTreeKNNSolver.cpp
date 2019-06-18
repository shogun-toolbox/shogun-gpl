/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/multiclass/CoverTreeKNNSolver.h>
#include <shogun/lib/JLCoverTree.h>

using namespace shogun;

CoverTreeKNNSolver::CoverTreeKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels):
KNNSolver(k, q, num_classes, min_label, train_labels) { /* nothing to do */ }

std::shared_ptr<MulticlassLabels> CoverTreeKNNSolver::classify_objects(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const
{
	auto output= std::make_shared<MulticlassLabels>(num_lab);

	// m_q != 1.0 not supported with cover tree because the neighbors
	// are not retrieved in increasing order of distance to the query
	if ( m_q != 1.0 )
		io::info("q != 1.0 not supported with cover tree, using q = 1");

	// From the sets of features (lhs and rhs) stored in distance,
	// build arrays of cover tree points
	v_array< JLCoverTreePoint > set_of_points  =
		parse_points(knn_distance, FC_LHS);
	v_array< JLCoverTreePoint > set_of_queries =
		parse_points(knn_distance, FC_RHS);

	// Build the cover trees, one for the test vectors (rhs features)
	// and another for the training vectors (lhs features)
	auto r = knn_distance->replace_rhs( knn_distance->get_lhs() );
	node<JLCoverTreePoint> top = batch_create(set_of_points);
	auto l = knn_distance->replace_lhs(r);
	knn_distance->replace_rhs(r);
	node<JLCoverTreePoint> top_query = batch_create(set_of_queries);

	// Get the k nearest neighbors to all the test vectors (batch method)
	knn_distance->replace_lhs(l);
	v_array< v_array< JLCoverTreePoint > > res;
	k_nearest_neighbor(top, top_query, res, m_k);

if (env()->io()->get_loglevel()<= io::MSG_DEBUG)
{
	SG_DEBUG("\nJL Results:")
	for ( int32_t i = 0 ; i < res.index ; ++i )
	{
		for ( int32_t j = 0 ; j < res[i].index ; ++j )
		{
			SG_DEBUG("{} ", res[i][j].m_index);
		}
		SG_DEBUG("");
	}
	SG_DEBUG("")
}

	for ( index_t i = 0 ; i < res.index ; ++i )
	{
		// Translate from indices to labels of the nearest neighbors
		for ( index_t j = 0; j < m_k; ++j )
			// The first index in res[i] points to the test vector
			train_lab[j] = m_train_labels.vector[ res[i][j+1].m_index ];

		// Get the index of the 'nearest' class
		index_t out_idx = choose_class(classes.vector, train_lab.vector);
		output->set_label(res[i][0].m_index, out_idx+m_min_label);
	}


	return output;
}

SGVector<int32_t> CoverTreeKNNSolver::classify_objects_k(std::shared_ptr<Distance> knn_distance, int32_t num_lab, SGVector<int32_t>& train_lab,  SGVector<int32_t>& classes) const
{
	SGVector<int32_t> output(m_k*num_lab);

	//allocation for distances to nearest neighbors
	SGVector<float64_t> dists(m_k);

	// From the sets of features (lhs and rhs) stored in distance,
	// build arrays of cover tree points
	v_array< JLCoverTreePoint > set_of_points  =
		parse_points(knn_distance, FC_LHS);
	v_array< JLCoverTreePoint > set_of_queries =
		parse_points(knn_distance, FC_RHS);

	// Build the cover trees, one for the test vectors (rhs features)
	// and another for the training vectors (lhs features)
	auto r = knn_distance->replace_rhs( knn_distance->get_lhs() );
	node< JLCoverTreePoint > top = batch_create(set_of_points);
	auto l = knn_distance->replace_lhs(r);
	knn_distance->replace_rhs(r);
	node< JLCoverTreePoint > top_query = batch_create(set_of_queries);

	// Get the k nearest neighbors to all the test vectors (batch method)
	knn_distance->replace_lhs(l);
	v_array< v_array< JLCoverTreePoint > > res;
	k_nearest_neighbor(top, top_query, res, m_k);

	for ( index_t i = 0 ; i < res.index ; ++i )
	{
		// Handle the fact that cover tree doesn't return neighbors
		// ordered by distance

		for ( index_t j = 0 ; j < m_k ; ++j )
		{
			// The first index in res[i] points to the test vector
			dists[j]     = knn_distance->distance(res[i][j+1].m_index,
						res[i][0].m_index);
			train_lab[j] = m_train_labels.vector[
						res[i][j+1].m_index ];
		}

		// Now we get the indices to the neighbors sorted by distance
		Math::qsort_index(dists.vector, train_lab.vector, m_k);

		choose_class_for_multiple_k(output.vector+res[i][0].m_index, classes.vector,
				train_lab.vector, num_lab);
	}

	return output;
}
