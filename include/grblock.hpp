#pragma once

#include <array>
#include <tuple>
#include <utility>
#include <algorithm>
#include <execution>
#include <type_traits>

using std::tuple;
using std::tuple_cat;
using std::tuple_size;
using std::make_tuple;

struct Covariant {};
struct Contravariant {};


template<size_t x, size_t n>
struct power {
    constexpr static size_t value = x * power<x,n-1>::value;
};
template<size_t x>
struct power<x,0> {
    constexpr static size_t value = 1;
};

template < typename T , typename... Ts >
auto head( std::tuple<T,Ts...> t )
{
   return  std::get<0>(t);
}

template < std::size_t... Ns , typename... Ts >
auto tail_impl( std::index_sequence<Ns...> , std::tuple<Ts...> t )
{
   return  std::make_tuple( std::get<Ns+1u>(t)... );
}

template < typename... Ts >
auto tail( std::tuple<Ts...> t )
{
   return  tail_impl( std::make_index_sequence<sizeof...(Ts) - 1u>() , t );
}

template<typename T, size_t N, typename ... Variances>
struct Tensor;

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



template<typename T, size_t N, typename ... Variances>
class Tensor {
public:
    typedef Tensor<T,N,Variances...> this_type;
    typedef T element_type;
    typedef std::array<T,TensorSize<N,Variances...>::value> data_type;
    typedef TensorHelper<T,N,sizeof...(Variances)> helper_type;
    typedef data_type::iterator iterator;
    typedef data_type::const_iterator const_iterator;
    // using index_type = TensorHelper<T,N,sizeof...(Variances)>::index_type;

    constexpr static size_t dimensions = N;
    constexpr static size_t degree = sizeof...(Variances);
    constexpr static size_t size() { return TensorSize<N,Variances...>::value; }

    Tensor();
    Tensor(data_type const & data);
    Tensor(data_type && data);
    Tensor(Tensor const &);
    Tensor(Tensor &&);
    this_type & operator=(this_type const &);
    this_type & operator=(this_type &&);
    ~Tensor();

    template<typename ... Sizes>
    size_t index(Sizes ... sizes) const { return helper_type::index(sizes...); }
    // size_t index()
    auto dimension(size_t index) const { return helper_type::dimension(index); }
    

    this_type & operator+=(this_type const &);
    this_type & operator-=(this_type const &);
    this_type & operator*=(T);
    this_type & operator/=(T);
    this_type operator+(this_type const &) const;
    this_type operator-(this_type const &) const;
    this_type operator*(T) const;
    this_type operator/(T) const;

    T & at(size_t index) { return data_[index]; }
    T const & at(size_t index) const { return data_[index]; }

    template<typename ... Sizes>
    T & get(Sizes ... sizes) { return data_.at(helper_type::index(sizes...)); }

    template<typename ... Sizes>
    T const & get(Sizes ... sizes) const { return data_.at(helper_type::index(sizes...)); }

    template<typename ... Sizes>
    T & operator()(Sizes ... sizes) { return data_.at(helper_type::index(sizes...)); }
    template<typename ... Sizes>
    T const & operator()(Sizes ... sizes) const { return data_.at(helper_type::index(sizes...)); }

    template<size_t i, size_t j>
    auto contract() const {
        return ContractionHelper<T, N, i, j, Variances...>::contract(*this);
    }

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

    void print(std::ostream & os) const;
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
Tensor<T,N,Variances...>::Tensor(data_type const & data) : data_(data) {}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::Tensor(data_type && data) : data_(std::move(data)) {}

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

template<typename T, size_t N, typename ... Variances>
void Tensor<T,N,Variances...>::print(std::ostream & os) const {
    PrintHelper<T,N,typename Tensor<T,N,Variances...>::const_iterator,Variances...>::print(os, data_.begin());
}
// template<typename T, size_t N, typename RandomAccessIterator, typename Variance, typename ... Variances>
// void PrintHelper<T,N,RandomAccessIterator,Variance,Variances...>::print_rows<0>(std::ostream & os, RandomAccessIterator i) {}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::~Tensor() {}

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

template<typename T, size_t N, typename ConractionType>
struct variances_to_tensor;

template<typename T, size_t N, typename ... Variances>
struct variances_to_tensor<T,N,variance_container<Variances...>> {
    typedef Tensor<T,N,Variances...> type;
};


template<typename Tuple, size_t ... Is>
auto tuple_head_impl(Tuple const & t, std::index_sequence<Is...>) {
    return make_tuple(get<Is>(t)...);
} 
template<size_t i, typename Tuple, size_t ... Is>
auto tuple_tail_impl(Tuple const & t, std::index_sequence<Is...>) {
    return make_tuple(get<i+Is>(t)...);
}

template<size_t i, typename ... Ts>
auto tuple_head(tuple<Ts...> const & t) {
    return tuple_head_impl(t, std::make_index_sequence<i>{});
}
template<size_t i, typename ... Ts>
auto tuple_tail(tuple<Ts...> const & t) {
    return tuple_tail_impl<i>(t, std::make_index_sequence<sizeof...(Ts)-i>{});
}

// template<size_t i, typename T, typename ... Ts>
// auto tuple_splice(tuple<Ts...> const & t, T const & val) {
//     return tuple_cat(
//         tuple_head<i>(t),
//         tuple<T>(val),
//         tuple_tail<i>(t)
//     );
// }
template<size_t i, typename T, typename ... Ts>
auto tuple_splice(tuple<Ts...> const & t, T val) {
    return tuple_cat(
        tuple_head<i>(t),
        tuple<T>(val),
        tuple_tail<i>(t)
    );
}


template<typename T, size_t N, size_t i, size_t j, typename ... Variances>
struct ContractionHelper {
    using type = typename variances_to_tensor<T,N,typename contraction_type<i,j,Variances...>::type>::type;

    static auto uncontract_indices(type::helper_type::index_type const & contracted) {
        return tuple_splice<(i<j?j:i)>(tuple_splice<(i<j?i:j)>(contracted, (T)0), (T)0);
    }

    // stride of this contraction in the original tensor
    constexpr static size_t stride() {
        // TensorHelper<T,N,sizeof...(Variances)>
        return power<N,i>::value + power<N,j>::value;
    }

    static type contract(Tensor<T,N,Variances...> const & t) 
    {
        type c;
        
        std::transform(std::execution::par_unseq, c.begin(), c.end(), c.begin(), [&](auto & element) {
            // first get the index
            auto index = &element - c.begin();

            // transform the index into tensor indices
            auto dims = c.dimension(index);

            // get the tensor index of the uncontracted tensor
            // t0100::helper_type::index_type uncontracted;
            // uncontract<0,1>(uncontracted, dims);
            auto un = uncontract_indices(dims);

            // get the index in the uncontracted
            // t.index(uncontracted);
            // auto undims = tuple_splice(dims)
            auto tdex = t.index(un);

            T dat = 0;
            for(size_t k = 0; k < N; ++k) {
                dat += t.at(tdex + k * stride());
            }

            return dat;
        });

        return c;
    }

};

template<typename Tensor, size_t i, size_t j>
struct Contraction;

template<typename T, size_t N, size_t i, size_t j, typename ... Variances>
struct Contraction<Tensor<T,N,Variances...>, i, j> : ContractionHelper<T,N,i,j,Variances...> {};


// template<typename T, size_t N, size_t i, size_t j, typename ... Variances>
// struct ContractionHelper {
//     void contract() {

//     }
// };


template<typename Number, size_t N>
struct GRBlock {

};
