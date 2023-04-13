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

