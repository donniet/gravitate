#pragma once

#include <array>
#include <tuple>
#include <algorithm>
#include <execution>
#include <type_traits>

using std::tuple;
using std::tuple_cat;
using std::make_tuple;

struct Covariant {};
struct Contravariant {};

template<typename T, size_t N, typename ... Variances>
struct Tensor;

template<typename Tensor, size_t i, size_t j>
struct ContractionHelper;

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

template<size_t N, typename ... Variances>
struct TensorSize;

template<typename T, size_t N, typename ... Variances>
class Tensor {
    typedef Tensor<T,N,Variances...> this_type;
    typedef std::array<T,TensorSize<N,Variances...>::value> data_type;
public:
    constexpr static size_t dimensions = N;
    constexpr static size_t degree = sizeof...(Variances);

    Tensor();
    Tensor(std::array<T,N*sizeof...(Variances)> const & data);
    Tensor(Tensor const &);
    Tensor(Tensor &&);
    this_type & operator=(this_type const &);
    this_type & operator=(this_type &&);
    ~Tensor();

    this_type & operator+=(this_type const &);
    this_type & operator-=(this_type const &);
    this_type & operator*=(T);
    this_type & operator/=(T);
    this_type operator+(this_type const &) const;
    this_type operator-(this_type const &) const;
    this_type operator*(T) const;
    this_type operator/(T) const;

    template<typename ... Sizes>
    T & get(Sizes ... sizes) { return data_.get(sizes...); }

    template<typename ... Sizes>
    T const & get(Sizes ... sizes) const { return data_.get(sizes...); }

    // template<size_t i, size_t j>
    // auto contract() const -> ContractionHelper<T, N, i, j, Variances...>::result_type;

    // template<typename ... SecondVariances>
    // auto operator*(Tensor<T,N,SecondVariances...> const & other) const -> 
    //     typename Tensor<T,N,Variances...,SecondVariances...> 
    // {
    //     Tensor<T,N,Variances...,SecondVariances...> ret(true);
            
    //     std::generate(std::execution::par_unseq, ret.data_.begin(), ret.data_.end(), [*this, &other]());
    // }

private:
    Tensor(bool) {}; // don't initialize data_
    data_type data_;

public:
    typename data_type::iterator begin() { return data_.begin(); }
    typename data_type::const_iterator begin() const { return data_.begin(); }
    typename data_type::iterator end() { return data_.end(); }
    typename data_type::const_iterator end() const { return data_.end(); }
};

// template<typename T, size_t N, size_t i, size_t j, typename ... Variances>
// auto Tensor<T,N,Variances...>::contract() const -> ContractionHelper<Tensor<T,N,Variances...>, i, j>::result_type {
//     return ContractionHelper<Tensor<T,N,Variances...>, i, j>::contract(*this);
// }

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::Tensor() { 
    std::fill(std::execution::par_unseq, data_.begin(), data_.end(), 0);
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::Tensor(Tensor<T,N,Variances...> const & other) {
    std::copy(std::execution::par_unseq, other.data_.begin(), other.data_.end(), data_.begin());
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::Tensor(Tensor<T,N,Variances...> && other) : data_(std::move(other.data_)) {}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...> & Tensor<T,N,Variances...>::operator=(Tensor<T,N,Variances...> const & other) {
    std::copy(std::execution::par_unseq, other.data_.begin(), other.data_.end(), data_.begin());
    return *this;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...> & Tensor<T,N,Variances...>::operator=(Tensor<T,N,Variances...> && other) {
    data_ = std::move(other.data_);
    return *this;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>& Tensor<T,N,Variances...>::operator+=(this_type const & other) {
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::plus<T>());
    return *this;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>& Tensor<T,N,Variances...>::operator-=(this_type const & other) {
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::minus<T>());
    return *this;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>& Tensor<T,N,Variances...>::operator*=(T scalar) {
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), data_.begin(), 
        [&scalar](T const & element) { return element * scalar; }
    );
    return *this;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>& Tensor<T,N,Variances...>::operator/=(T scalar) {
    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), data_.begin(), 
        [&scalar](T const & element) { return element / scalar; }
    );
    return *this;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...> Tensor<T,N,Variances...>::operator+(Tensor<T,N,Variances...> const & other) const {
    Tensor<T,N,Variances...> result(true); // uninitialized

    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), 
        std::plus<T>()
    );
    return result;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...> Tensor<T,N,Variances...>::operator-(Tensor<T,N,Variances...> const & other) const {
    Tensor<T,N,Variances...> result(true); // uninitialized

    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), 
        std::minus<T>()
    );
    return result;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...> Tensor<T,N,Variances...>::operator*(T scalar) const {
    Tensor<T,N,Variances...> result(true); // uninitialized

    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), result.data_.begin(), 
        [&scalar](T const & element) { return element * scalar; }
    );
    return result;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...> Tensor<T,N,Variances...>::operator/(T scalar) const {
    Tensor<T,N,Variances...> result(true); // uninitialized

    std::transform(std::execution::par_unseq, data_.begin(), data_.end(), result.data_.begin(), 
        [&scalar](T const & element) { return element / scalar; }
    );
    return result;
}

template<typename T, size_t N>
struct TensorHelper<T,N,0> {
    static size_t index() { return 0; }
};
template<typename T, size_t N>
struct TensorHelper<T,N,1> {
    typedef tuple<T> dimension_type;

    static size_t index(size_t i) { return i; }
    static dimension_type dimension(size_t index) {
        return {(T)index};
    }
};
template<typename T, size_t N, size_t M>
struct TensorHelper {
    template<typename ... Sizes>
    static size_t index(size_t i, Sizes ... sizes) {
        return i + N * TensorHelper<T,N,M-1>::index(sizes...);
    }
    static auto dimension(size_t index) ->
        decltype(tuple_cat(make_tuple<T>(index%N),TensorHelper<T,N,M-1>::dimension(index/N)))    
    {
        return tuple_cat(make_tuple<T>(index % N), 
            TensorHelper<T,N,M-1>::dimension(index / N)
        );
    }
};




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


// template<typename T, size_t N, size_t i, size_t j, typename ... Variances>
// struct ContractionHelper {
//     void contract() {

//     }
// };


template<typename Number, size_t N>
struct GRBlock {

};
