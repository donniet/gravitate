#pragma once

#include <array>
#include <tuple>

using std::tuple;
using std::tuple_cat;
using std::make_tuple;

struct Covariant {};
struct Contravariant {};

template<typename T, size_t N, typename ... Variances>
struct Tensor;

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

template<typename T, size_t N, typename ... Variances>
class Tensor {
    typedef Tensor<T,N,Variances...> this_type;
public:
    Tensor();
    Tensor(std::array<T,N*sizeof...(Variances)> const & data);
    Tensor(Tensor const &);
    Tensor(Tensor &&);
    this_type & operator=(Tensor const &);
    this_type & operator=(Tensor &&);
    ~Tensor();

    this_type & operator+=(Tensor const &);
    this_type & operator-=(Tensor const &);
    this_type & operator*=(T);
    this_type & operator/=(T);
    this_type operator+(Tensor const &) const;
    this_type operator-(Tensor const &) const;
    this_type operator*(T) const;

private:
    std::array<T,N*sizeof...(Variances)> data_;
};

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






template<typename Number, size_t N>
struct GRBlock {

};
