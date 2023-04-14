#pragma once


#include "tuple_splice.hpp"
#include "detail/tensor_detail.hpp"

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

template<typename ... Variances>
struct index_type;


template<> struct index_type<> { typedef tuple<> type; };

template<typename Variance> struct index_type<Variance> {
    typedef tuple<size_t> type;
};

template<typename Variance, typename ... Variances>
struct index_type<Variance, Variances...> {
    typedef decltype(tuple_cat(tuple<size_t>(), typename index_type<Variances...>::type())) type;
};

template<typename T, size_t N, typename ... Variances>
class Tensor;

template<typename T, size_t N, typename ConractionType>
struct variances_to_tensor;

template<typename T, size_t N, typename ... Variances>
struct variances_to_tensor<T,N,variance_container<Variances...>> {
    typedef Tensor<T,N,Variances...> type;
};

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
    Tensor(bool) {}; // don't initialize data_
    Tensor(data_type const & data);
    Tensor(data_type && data);
    Tensor(Tensor const &);
    Tensor(Tensor &&);
    this_type & operator=(this_type const &);
    this_type & operator=(this_type &&);
    ~Tensor();

    template<typename ... Sizes>
    static size_t index(Sizes ... sizes) { return helper_type::index(sizes...); }
    // size_t index()
    static auto dimension(size_t index) { return helper_type::dimension(index); }
    

    this_type & operator+=(this_type const &);
    this_type & operator-=(this_type const &);
    this_type & operator*=(T);
    this_type & operator/=(T);
    this_type operator+(this_type const &) const;
    this_type operator-(this_type const &) const;
    this_type operator*(T) const;

    template<typename ... SecondVariances>
    Tensor<T,N,Variances...,SecondVariances...> operator*(Tensor<T,N,SecondVariances...> const & other) const;

    this_type operator/(T) const;

    T & at(size_t index) { return data_[index]; }
    T const & at(size_t index) const { return data_[index]; }

    template<typename ... Sizes>
    T & get(Sizes ... sizes) { return data_.at(helper_type::index(sizes...)); }

    template<typename ... Sizes>
    T const & get(Sizes ... sizes) const { return data_.at(helper_type::index(sizes...)); }

    T & in(typename index_type<Variances...>::type index) {
        return data_.at(helper_type::index(index));
    }
    T const & in(typename index_type<Variances...>::type index) const {
        return data_.at(helper_type::index(index));
    }

    template<typename ... Sizes>
    T & operator()(Sizes ... sizes) { return data_.at(helper_type::index(sizes...)); }
    template<typename ... Sizes>
    T const & operator()(Sizes ... sizes) const { return data_.at(helper_type::index(sizes...)); }

    // T & operator()(typename index_type<Variances...>::type index) { return get(index); }
    // T const & operator()(typename index_type<Variances...>::type index) const { return get(index); }


    template<size_t i, size_t j>
    auto contract() const;

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
    data_type data_;

    // stride of this contraction in the original tensor
    template<size_t i, size_t j>
    constexpr static size_t stride() {
        // TensorHelper<T,N,sizeof...(Variances)>
        return power<N,i>::value + power<N,j>::value;
    }

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
template<typename ... SecondVariances>
Tensor<T,N,Variances...,SecondVariances...> Tensor<T,N,Variances...>::operator*(Tensor<T,N,SecondVariances...> const & other) const {
    typedef Tensor<T,N,Variances...,SecondVariances...> result_type;

    result_type ret(true); // uninitialized

    auto beg = ret.begin();

    std::transform(std::execution::par_unseq, beg, ret.end(), beg, [&](T const & element) -> T {
        // get the offset of this element
        size_t dex = &element - beg;

        // turn the dex into a tuple of coordinates
        auto c = result_type::dimension(dex);

        // split the dimensions into the first and second coordinates
        auto ac = tuple_head<sizeof...(Variances)>(c);
        auto bc = tuple_tail<sizeof...(Variances)>(c);

        // return the multiple of the elemtents from both tensors
        return in(ac) * other.in(bc);
    });

    return ret;
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


template<typename T, size_t N, typename ... Variances>
void Tensor<T,N,Variances...>::print(std::ostream & os) const {
    PrintHelper<T,N,typename Tensor<T,N,Variances...>::const_iterator,Variances...>::print(os, data_.begin());
}
// template<typename T, size_t N, typename RandomAccessIterator, typename Variance, typename ... Variances>
// void PrintHelper<T,N,RandomAccessIterator,Variance,Variances...>::print_rows<0>(std::ostream & os, RandomAccessIterator i) {}

template<size_t i, size_t j, typename T, typename ... Ts>
auto uncontract_indices(tuple<Ts...> const & contracted, T val) {
    return tuple_splice<(i<j?j:i)>(tuple_splice<(i<j?i:j)>(contracted, val), val);
}

template<typename T, size_t N, typename ... Variances>
template<size_t i, size_t j>
auto Tensor<T,N,Variances...>::contract() const {
    typedef typename variances_to_tensor<T,N,typename contraction_type<i,j,Variances...>::type>::type contracted_type;
    contracted_type c;
        
    std::transform(std::execution::par_unseq, c.begin(), c.end(), c.begin(), [&](auto & element) {
        // first get the index
        auto dex = &element - c.begin();

        // transform the index into tensor indices
        auto dims = c.dimension(dex);

        // get the tensor index of the uncontracted tensor
        // t0100::helper_type::index_type uncontracted;
        // uncontract<0,1>(uncontracted, dims);
        auto un = uncontract_indices<i,j>(dims, T(0));

        // get the index in the uncontracted
        // t.index(uncontracted);
        // auto undims = tuple_splice(dims)
        auto tdex = index(un);

        T dat = 0;
        for(size_t k = 0; k < N; ++k) {
            dat += at(tdex + k * stride<i,j>());
        }

        return dat;
    });

    return c;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::~Tensor() {}
