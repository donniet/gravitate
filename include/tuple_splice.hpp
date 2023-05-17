#pragma once

#include <cstddef>
#include <tuple>

using std::tuple;
using std::get;
using std::make_tuple;

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

template<size_t i, typename T, typename ... Ts>
auto tuple_splice(tuple<Ts...> const & t, T val) {
    return tuple_cat(
        tuple_head<i>(t),
        tuple<T>(val),
        tuple_tail<i>(t)
    );
}

template<typename Seq0, typename Seq1>
struct sequence_join;

template<size_t ... Is, size_t ... Js>
struct sequence_join<std::index_sequence<Is...>, std::index_sequence<Js...>> {
    typedef std::index_sequence<Is..., Js...> type;
};

template<size_t Remove, size_t ... Is>
struct remove_sequence_element;

template<size_t Remove>
struct remove_sequence_element<Remove> {
    typedef std::index_sequence<> type;
};

template<size_t Remove, size_t First, size_t ... Rest>
struct remove_sequence_element<Remove, First, Rest...> {
    typedef typename std::conditional_t<Remove == First,
        typename remove_sequence_element<Remove, Rest...>::type,
        typename sequence_join<
            std::index_sequence<First>, 
            typename remove_sequence_element<Remove, Rest...>::type
        >::type
    > type;
};

template<size_t Exists, size_t ... Is>
struct element_exists {
    static constexpr bool value = false;
};

template<size_t Exists, size_t First, size_t ... Rest>
struct element_exists<Exists, First, Rest...> {
    static constexpr bool value = Exists == First || element_exists<Exists, Rest...>::value;
};

template<typename Seq> struct is_permutation;

template<> struct is_permutation<std::index_sequence<>> { 
    static constexpr bool value = true;
};

template<size_t ... Is>
struct is_permutation<std::index_sequence<Is...>> {
    static constexpr bool value = 
        element_exists<sizeof...(Is)-1, Is...>::value &&
        is_permutation<typename remove_sequence_element<sizeof...(Is)-1, Is...>::type>::value;
};

template<typename Seq, typename Enable = void>
struct permutation_helper;

template<size_t ... Is>
struct permutation_helper<std::index_sequence<Is...>, 
    typename std::enable_if<
        is_permutation<
            std::index_sequence<Is...>
        >::value
    >::type> 
{};

template<size_t ... Is>
struct permutation : public permutation_helper<std::index_sequence<Is...>> {
    template<typename T> static constexpr auto permute(T const &);    
};

// template<size_t ... Is>
// template<typename ... Ts>
// auto permutation<Is...>::permute(tuple<Ts...> const & tup) {

// }






