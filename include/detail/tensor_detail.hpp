#pragma once

#include <cstddef>
#include <tuple>
#include <iostream>

using std::tuple;
using std::make_tuple;

struct Covariant;
struct Contravariant;

template<size_t x, size_t n>
struct power {
    constexpr static size_t value = x * power<x,n-1>::value;
};
template<size_t x>
struct power<x,0> {
    constexpr static size_t value = 1;
};


template<size_t N, typename ... Variances>
struct TensorSize;

template<typename T, size_t N, size_t i, size_t j, typename ... Variances>
struct ContractionHelper;

template<size_t N>
struct TensorSize<N> {
    static constexpr size_t value = 1;
};

template<size_t N, typename Variance, typename ... Variances>
struct TensorSize<N, Variance, Variances...> {
    static constexpr size_t value = N * TensorSize<N, Variances...>::value;
};

template<size_t i, typename ... Variances>
struct is_contravariant {
    static constexpr bool value = false;
};
template<typename ... Variances>
struct is_contravariant<0, Contravariant, Variances...> {
    static constexpr bool value = true;
};
template<typename ... Variances>
struct is_contravariant<0, Covariant, Variances...> {
    static constexpr bool value = false;
};
template<size_t i, typename Variance, typename ... Variances>
struct is_contravariant<i, Variance, Variances...> {
    static constexpr bool value = is_contravariant<i-1, Variances...>::value;
};

template<size_t i, typename ... Variances>
struct is_covariant;

template<size_t i, typename ... Variances>
struct is_covariant {
    static constexpr bool value = false;
};
template<typename ... Variances>
struct is_covariant<0, Covariant, Variances...> {
    static constexpr bool value = true;
};
template<typename ... Variances>
struct is_covariant<0, Contravariant, Variances...> {
    static constexpr bool value = false;
};
template<size_t i, typename Variance, typename ... Variances>
struct is_covariant<i, Variance, Variances...> {
    static constexpr bool value = is_covariant<i-1, Variances...>::value;
};

template<size_t i, size_t j, typename ... Variances>
struct contraction_type;

/*
this is the code to verify if a contraction works at compile time.
std::enable_if_t<
std::disjunction<
    std::conjunction<
        is_contravariant<i, Variance, Variances...>,
        is_covariant<j, Variance, Variances...>
    >,
    std::conjunction<
        is_covariant<i, Variance, Variances...>,
        is_contravariant<j, Variance, Variances...>
    >
>::value>
*/

template<typename ... Variances>
struct variance_container {};

template<typename Variance, typename Tensor>
struct tensor_cat;

template<size_t i, size_t j, typename ... Variances>
struct contraction_type_inner;

template<size_t j, typename ... Variances>
struct contraction_type_covariant_first;

template<size_t j, typename ... Variances>
struct contraction_type_contravariant_first;

template<size_t covariant_index, size_t contravariant_index>
struct contraction_indicies;

template<size_t i, size_t j>
struct max_v {
    constexpr static size_t value = i > j ? i : j;
};

template<size_t i, size_t j>
struct min_v {
    constexpr static size_t value = i > j ? j : i;
};

template<size_t i, size_t j, typename ... Variances>
struct contraction_type {
    using type = contraction_type_inner<min_v<i,j>::value, max_v<i,j>::value, Variances...>::type;
};


template<size_t j, typename Variance, typename ... Variances>
struct contraction_type_covariant_first<j, Variance, Variances...>
{
    using type = tensor_cat<
        Variance,
        typename contraction_type_covariant_first<j-1, Variances...>::type
    >::type;
};
template<typename ... Variances>
struct contraction_type_covariant_first<0, Contravariant, Variances...>
{
    using type = variance_container<Variances...>;
};

template<size_t j, typename Variance, typename ... Variances>
struct contraction_type_contravariant_first<j, Variance, Variances...>
{
    using type = tensor_cat<
        Variance,
        typename contraction_type_contravariant_first<j-1, Variances...>::type
    >::type;
};
template<typename ... Variances>
struct contraction_type_contravariant_first<0, Covariant, Variances...>
{
    using type = variance_container<Variances...>;
};

template<size_t i, size_t j, typename Variance, typename ... Variances>
struct contraction_type_inner<i, j, Variance, Variances...> 
{
    using type = tensor_cat<
        Variance,
        typename contraction_type_inner<i-1, j-1, Variances...>::type
    >::type;
};
template<size_t j, typename ... Variances>
struct contraction_type_inner<0, j, Covariant, Variances...>
{
    using type = contraction_type_covariant_first<j-1, Variances...>::type;
};
template<size_t j, typename ... Variances>
struct contraction_type_inner<0, j, Contravariant, Variances...>
{
    using type = contraction_type_contravariant_first<j-1, Variances...>::type;
};


template<typename Variance, typename ... Variances>
struct tensor_cat<Variance, variance_container<Variances...>> {
    using type = variance_container<Variance,Variances...>;
};


template<typename T, size_t N, size_t M>
struct TensorHelper;

template<typename ... Variances>
struct VariancesHelper;

template<>
struct VariancesHelper<Contravariant> {
    static tuple<bool> contravariant() { return make_tuple(true); }
};

template<>
struct VariancesHelper<Covariant> {
    static tuple<bool> contravariant() { return make_tuple(false); }
};

template<typename ... Variances>
struct VariancesHelper<Contravariant, Variances...> {
    static auto contravariant() -> decltype(tuple_cat(tuple<bool>(true),
                                            VariancesHelper<Variances...>::contravariant())) {
        return tuple_cat(tuple<bool>(true),
                         VariancesHelper<Variances...>::contravariant());
    }
};

template<typename ... Variances>
struct VariancesHelper<Covariant, Variances...> {
    static auto contravariant() -> decltype(tuple_cat(tuple<bool>(true),
                                            VariancesHelper<Variances...>::contravariant())) {
        return tuple_cat(tuple<bool>(true),
                         VariancesHelper<Variances...>::contravariant());
    }
};

template<typename ... Variances>
struct IndexHelper;


template<typename T, size_t N, typename RandomAccessIterator, typename ... Variances>
struct PrintHelper;

//TODO: I believe this prints in the wrong major order, 
// but it does print all elements in the proper structure, 
// so we will leave it be for now.
template<typename T, size_t N, typename RandomAccessIterator, typename ... Variances>
struct PrintHelper<T,N,RandomAccessIterator,Covariant,Variances...> {
    static void print(std::ostream & os, RandomAccessIterator i) {
        os << "[";
        if(sizeof...(Variances) == 0) {
            os << " ";
        }
        for(int j = 0; j < N; ++j) {
            PrintHelper<T,N,RandomAccessIterator,Variances...>::print(os, i + j * power<N,sizeof...(Variances)>::value);
        }
        os << "] ";
    }
};
template<typename T, size_t N, typename RandomAccessIterator, typename ... Variances>
struct PrintHelper<T,N,RandomAccessIterator,Contravariant,Variances...> {
    static void print(std::ostream & os, RandomAccessIterator i) {
        os << "{";
        if(sizeof...(Variances) == 0) {
            os << " ";
        }
        for(int j = 0; j < N; ++j) {
            PrintHelper<T,N,RandomAccessIterator,Variances...>::print(os, i + j * power<N,sizeof...(Variances)>::value);
        }
        os << "} ";
    }
};

template<typename T, size_t N, typename RandomAccessIterator>
struct PrintHelper<T,N,RandomAccessIterator> {
    static void print(std::ostream & os, RandomAccessIterator i) {
        os << *i << " ";
    }
};


template<typename T, size_t N>
struct TensorHelper<T,N,0> {
    typedef tuple<> dimension_type;
    static size_t index(size_t i) { return i; }
    static dimension_type dimension(size_t index) { return tuple<>{}; }
    typedef decltype(dimension(0)) index_type;
};
template<typename T, size_t N>
struct TensorHelper<T,N,1> {
    typedef tuple<T> dimension_type;

    static size_t index(size_t i) { return i; }
    static size_t index(tuple<size_t> ti) { return get<0>(ti); }
    static dimension_type dimension(size_t index) {
        return {(T)index};
    }
    typedef decltype(dimension(0)) index_type;
};
template<typename T, size_t N, size_t M>
struct TensorHelper {
    template<typename ... Sizes>
    static size_t index(size_t i, Sizes ... sizes) {
        return i + N * TensorHelper<T,N,M-1>::index(sizes...);
    }
    template<typename ... Sizes>
    static size_t index(tuple<Sizes...> sizes);
    template<typename ... Sizes>
    static size_t index_helper(size_t i, tuple<Sizes...> sizes);

    static auto dimension(size_t index)   
    {
        return tuple_cat(make_tuple<T>(index % N), 
            TensorHelper<T,N,M-1>::dimension(index / N)
        );
    }

    typedef decltype(dimension(0)) index_type;
};


template<typename T, size_t N, size_t M>
template<typename ... Sizes>
size_t TensorHelper<T,N,M>::index(tuple<Sizes...> sizes) {
    return index(head(sizes), tail(sizes));
}
template<typename T, size_t N, size_t M>
template<typename ... Sizes>
size_t TensorHelper<T,N,M>::index_helper(size_t i, tuple<Sizes...> sizes) {
    return i + N * TensorHelper<T,N,M-1>::index(sizes);
}




