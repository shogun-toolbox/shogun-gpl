/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) John Langford and Dinoj Surendran, v_array and its templatization
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _JLCTPOINT_H__
#define _JLCTPOINT_H__

#include <shogun/lib/config.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>

namespace shogun
{

namespace detail
{

/** @brief Class v_array taken directly from JL's implementation */
template<class T>
class v_array{

	public:
		/** Getter for the the last element of the v_array
		 *  @return the last element of the array */
		T last() { return elements[index-1];}

		/** Decrement the pointer to the last element */
		void decr() { index--;}

		/** Create an empty v_array */
		v_array() { index = 0; length=0; elements = NULL;}

		/** Element access operator
		 *  @param i of the element to be read
		 *  @return the corresponding element */
		T& operator[](unsigned int i) { return elements[i]; }

	public:
		/** Pointer to the last element of the v_array */
		int index;

		/** Length of the v_array */
		int length;

		/** Pointer to the beginning of the v_array elements */
		T* elements;

};

} // namespace detail

/**
 * Insert a new element at the end of the vector
 *
 * @param v vector
 * @param new_ele element to insert
 */
template<class T>
void push(detail::v_array<T>& v, const T &new_ele)
{
	auto old_len = v.length;
	while(v.index >= v.length)
	{
		v.length = 2*v.length + 3;
		v.elements = (T *)SG_REALLOC(T, v.elements, old_len, v.length);
	}
	v[v.index++] = new_ele;
}

/**
 * Used to modify the capacity of the vector
 *
 * @param v vector
 * @param length the new length of the vector
 */
template<class T>
void alloc(detail::v_array<T>& v, int length)
{
	v.elements = (T *)SG_REALLOC(T, v.elements, v.length, length);
	v.length = length;
}

/**
 * Returns the vector previous to the pointed one in the stack of
 * vectors and decrements the index of the stack. No memory is
 * freed here. If there are no vectors stored in the stack, create
 * and return a new empty vector
 *
 * @param stack of vectors
 * @return the adequate vector according to the previous conditions
 */
template<class T>
detail::v_array<T> pop(detail::v_array<detail::v_array<T> > &stack)
{
	if (stack.index > 0)
		return stack[--stack.index];
	else
		return detail::v_array<T>();
}

/**
 * Type used to indicate where to find (either lhs or rhs) the
 * coordinate information  of this point in the Distance object
 * associated
 */
enum EFeaturesContainer
{
	FC_LHS = 0,
	FC_RHS = 1,
};

/** @brief Class Point to use with John Langford's CoverTree. This
 * class must have some assoficated functions defined (distance,
 * parse_points and print, see below) so it can be used with the
 * CoverTree implementation.
 */
class JLCoverTreePoint
{

	public:

		/** Distance object where to find the coordinate information of
		 * this point */
		std::shared_ptr<Distance> m_distance;

		/** Index of this point in m_distance */
		int32_t m_index;

		/** If the point is stored in rhs or lhs in m_distance */
		EFeaturesContainer m_features_container;

}; /* class JLCoverTreePoint */

/** Functions declared out of the class definition to respect JLCoverTree
 *  structure */

float distance(JLCoverTreePoint p1, JLCoverTreePoint p2, float64_t upper_bound)
{
	/** Call m_distance->distance() with the proper index order depending on
	 *  the feature containers in m_distance for each of the points*/

	if ( p1.m_features_container == p2.m_features_container )
	{
		if ( ! p1.m_distance->lhs_equals_rhs() )
		{
			error("lhs != rhs but the distance of two points "
			       "from the same container has been requested");
		}
		else
		{
			return p1.m_distance->distance_upper_bounded(p1.m_index,
					p2.m_index, upper_bound);
		}
	}
	else
	{
		if ( p1.m_distance->lhs_equals_rhs() )
		{
			error("lhs == rhs but the distance of two points "
			      "from different containers has been requested");
		}
		else
		{
			if ( p1.m_features_container == FC_LHS )
			{
				return p1.m_distance->distance_upper_bounded(p1.m_index,
						p2.m_index, upper_bound);
			}
			else
			{
				return p1.m_distance->distance_upper_bounded(p2.m_index,
						p1.m_index, upper_bound);
			}
		}
	}

	error("Something has gone wrong, case not handled");
	return -1;
}

/** Fills up a v_array of JLCoverTreePoint objects */
detail::v_array< JLCoverTreePoint > parse_points(std::shared_ptr<Distance> distance, EFeaturesContainer fc)
{
	std::shared_ptr<Features> features;
	if ( fc == FC_LHS )
		features = distance->get_lhs();
	else
		features = distance->get_rhs();

	detail::v_array< JLCoverTreePoint > parsed;
	for ( int32_t i = 0 ; i < features->get_num_vectors() ; ++i )
	{
		JLCoverTreePoint new_point;

		new_point.m_distance = distance;
		new_point.m_index = i;
		new_point.m_features_container = fc;

		push(parsed, new_point);
	}

	return parsed;
}

/** Print the information of the CoverTree point */
void print(JLCoverTreePoint &p)
{
	error("Print JLCoverTreePoint not implemented");
}

} /* namespace shogun */

#endif /* _JLCTPOINT_H__*/
