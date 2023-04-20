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

template<typename T, typename R>
struct replace_type {
    typedef R type;
};

template<typename ... Variances>
struct index_type {
    typedef tuple<typename replace_type<Variances, size_t>::type...> type;
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
    static auto dimension(size_t index) { return helper_type::dimension(index); }
    
    bool operator==(this_type const &) const;
    bool operator!=(this_type const & other) const { return !(*this == other); }

    this_type & operator+=(this_type const &);
    this_type & operator-=(this_type const &);
    this_type & operator*=(T);
    this_type & operator/=(T);
    this_type operator+(this_type const &) const;
    this_type operator-(this_type const &) const;
    this_type operator*(T) const;
    this_type operator/(T) const;

    template<typename ... SecondVariances>
    Tensor<T,N,Variances...,SecondVariances...> operator*(Tensor<T,N,SecondVariances...> const & other) const;

    template<size_t i, size_t j, typename ... SecondVariances>
    typename variances_to_tensor<T,N,typename contraction_type<i,j,Variances...,SecondVariances...>::type>::type
    multiplyAndContract(Tensor<T,N,SecondVariances...> const & other) const;

    T & operator[](size_t index) { return data_[index]; }
    T const & operator[](size_t index) const { return data_[index]; }
    T & at(size_t index) { return data_.at(index); }
    T const & at(size_t index) const { return data_.at(index); }

    T & get(typename index_type<Variances...>::type index) {
        return data_.at(helper_type::index(index));
    }
    T const & get(typename index_type<Variances...>::type index) const {
        return data_.at(helper_type::index(index));
    }

    T & operator()(typename index_type<Variances...>::type index) { return get(index); }
    T const & operator()(typename index_type<Variances...>::type index) const { return get(index); }

    template<size_t i, size_t j>
    auto contract() const;

private:
    data_type data_;

    // stride of this contraction in the original tensor
    // i realize that's confusion
    //TODO: figure out a better place for this static function, maybe the contraction helper?
    // although it only depends externally on N...  Maybe we just take it out of this class altogether?
    template<size_t i, size_t j>
    constexpr static size_t stride() {
        return power<N,i>::value + power<N,j>::value;
    }

public:
    typename data_type::iterator begin() { return data_.begin(); }
    typename data_type::const_iterator begin() const { return data_.begin(); }
    typename data_type::iterator end() { return data_.end(); }
    typename data_type::const_iterator end() const { return data_.end(); }

    void print(std::ostream & os) const;
};

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
bool Tensor<T,N,Variances...>::operator==(Tensor<T,N,Variances...> const & other) const {
    return std::equal(std::execution::par_unseq, data_.begin(), data_.end(), other.data_.begin());
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

template<typename Tuple, size_t ... Is>
void print_tuple(std::ostream & os, Tuple const & t, std::index_sequence<Is...>) {
    os << "(";
    ((os << (Is == 0? "" : ", ") << std::get<Is>(t)), ...);
    os << ")";
}

template<typename ... Ts>
std::ostream & operator<<(std::ostream & os, tuple<Ts...> const & t) {
    print_tuple(os, t, std::make_index_sequence<sizeof...(Ts)>{});
    return os;
}

template<typename T, size_t N, typename ... Variances>
template<size_t i, size_t j, typename ... SecondVariances>
typename variances_to_tensor<T,N,typename contraction_type<i,j,Variances...,SecondVariances...>::type>::type
Tensor<T,N,Variances...>::multiplyAndContract(Tensor<T,N,SecondVariances...> const & other) const {
    typedef Tensor<T,N,Variances...,SecondVariances...> product_type;
    typedef typename variances_to_tensor<T,N,typename contraction_type<i,j,Variances...,SecondVariances...>::type>::type result_type;

    result_type ret(true); // uninitialized

    auto beg = ret.begin();

    std::transform(std::execution::par_unseq, beg, ret.end(), beg, [&](T const & element) -> T {
        // get the offset of this element
        size_t dex = &element - beg;

        T dat = 0;

        for(size_t k = 0; k < N; ++k, dex += stride<i,j>()) {
            // turn the offset into tensor indices of the product
            auto dims = product_type::dimension(dex);
            std::cerr << "dims: " << dims << std::endl;

            // split the dimensions into the first and second indices
            auto ac = tuple_head<sizeof...(Variances)>(dims);
            auto bc = tuple_tail<sizeof...(Variances)>(dims);

            // add the product to our result
            dat += get(ac) * other.get(bc);
        }
        return dat;
    });

    return ret;
}

template<typename T, size_t N, typename ... Variances>
template<typename ... SecondVariances>
Tensor<T,N,Variances...,SecondVariances...> 
Tensor<T,N,Variances...>::operator*(Tensor<T,N,SecondVariances...> const & other) const {
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
        return get(ac) * other.get(bc);
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
            dat += operator[](tdex + k * stride<i,j>());
        }

        return dat;
    });

    return c;
}

template<typename T, size_t N, typename ... Variances>
Tensor<T,N,Variances...>::~Tensor() {}
