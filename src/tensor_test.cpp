#include "tensor.hpp"

#include <filesystem>
#include <iostream>
#include <functional>
#include <array>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>


TEST(TensorTest, Stride) {
    // ASSERT_EQ(tensor_stride<4,0,1>)
}

TEST(TensorTest, Tuples) {
    tuple<int, int, int> a(0,1,3);
    auto head = tuple_head<2>(a);
    static_assert(std::tuple_size<decltype(head)>::value == 2);
    ASSERT_EQ(get<0>(head), 0);
    ASSERT_EQ(get<1>(head), 1);

    auto tail = tuple_tail<2>(a);
    static_assert(std::tuple_size<decltype(tail)>::value == 1);
    ASSERT_EQ(get<0>(tail), 3);

    auto full = tuple_splice<2>(a, 2);
    static_assert(std::tuple_size<decltype(full)>::value == 4);
    ASSERT_EQ(get<0>(full), 0);
    ASSERT_EQ(get<1>(full), 1);
    ASSERT_EQ(get<2>(full), 2);
    ASSERT_EQ(get<3>(full), 3);

    auto full_tail = tuple_tail<1>(full);
    static_assert(std::tuple_size<decltype(full_tail)>::value == 3);

    auto full2 = tuple_splice<0>(a, tuple_splice<2>(a, 2));

    // static_assert(is_permutation<std::index_sequence<1,0>>::value);
    static_assert(element_exists<0,0>::value);
    static_assert(element_exists<0,0,1>::value);
    static_assert(element_exists<0,1,0>::value);

    static_assert(element_exists<1,0,1>::value);
    static_assert(element_exists<1,1,0>::value);
    static_assert(!element_exists<2,0,1>::value);

    static_assert(std::is_same_v<
        remove_sequence_element<3, 0,1,2,3,4 >::type,
        std::index_sequence<0,1,2,4>
    >);

    static_assert(std::is_same_v<
        remove_sequence_element<0, 0,1,2,3,4 >::type,
        std::index_sequence<1,2,3,4>
    >);

    static_assert(is_permutation<std::index_sequence<0,1,2>>::value);
    #if 0 // this fails because 0,1,1 is not a permutation
    static_assert(is_permutation<std::index_sequence<0,1,1>>::value);   
    #endif

    permutation<0,1,2> p0;
    permutation<0,2,1> p1;
    permutation<1,0,2> p2;
    permutation<1,2,0> p3;
    permutation<2,0,1> p4;
    permutation<2,1,0> p5;

    #if 0 // this fails
    permutation<0,1,1> ERROR;
    #endif
}


TEST(TensorTest, Mult) {
    Tensor<float,2,Covariant>     a({2,3});
    Tensor<float,2,Contravariant> b({5,7});

    auto res = a * b;

    a.print(std::cout);
    std::cout << std::endl;
    b.print(std::cout);
    std::cout << std::endl;
    res.print(std::cout);
    std::cout << std::endl;

    res.print(std::cout);
    std::cout << std::endl;

    // check all the results are right (see above)
    ASSERT_EQ(res, (Tensor<float,2,Covariant,Contravariant>({10,15,14,21})));

    // check that our multiplication indexing works
    ASSERT_EQ(a({0}) * b({0}), res({0,0}));
    ASSERT_EQ(a({0}) * b({1}), res({0,1}));
    ASSERT_EQ(a({1}) * b({0}), res({1,0}));
    ASSERT_EQ(a({1}) * b({1}), res({1,1}));
}


TEST(TensorTest, Mult3) {
    Tensor<float,3,Covariant>     a({2, 3, 5});
    Tensor<float,3,Contravariant> b({7,11,13});

    auto res = a * b;

    a.print(std::cout);
    std::cout << std::endl;
    b.print(std::cout);
    std::cout << std::endl;
    res.print(std::cout);
    std::cout << std::endl;

    res.print(std::cout);
    std::cout << std::endl;

    // check all the results are right (see above)
    ASSERT_EQ(res, (Tensor<float,3,Covariant,Contravariant>({
        14,21,35, 
        22,33,55, 
        26,39,65
    })));

    // check that our multiplication indexing works
    ASSERT_EQ(a({0}) * b({0}), res({0,0}));
    ASSERT_EQ(a({0}) * b({1}), res({0,1}));
    ASSERT_EQ(a({0}) * b({2}), res({0,2}));
    ASSERT_EQ(a({1}) * b({0}), res({1,0}));
    ASSERT_EQ(a({1}) * b({1}), res({1,1}));
    ASSERT_EQ(a({1}) * b({2}), res({1,2}));
    ASSERT_EQ(a({2}) * b({0}), res({2,0}));
    ASSERT_EQ(a({2}) * b({1}), res({2,1}));
    ASSERT_EQ(a({2}) * b({2}), res({2,2}));
}

TEST(TensorTest, Mult4) {
    Tensor<float,4,Covariant>     a({2, 3, 5, 7});
    Tensor<float,4,Contravariant> b({11,13,15,17});
    Tensor<float,4> r;
    r[0] = 2 * 11 + 3 * 13 + 5 * 15 + 7 * 17;
    
    auto res = a.multiplyAndContract<0,1>(b);

    ASSERT_EQ(res, r);
    ASSERT_EQ(res({}), 2 * 11 + 3 * 13 + 5 * 15 + 7 * 17);

    ASSERT_EQ(res, ((a * b).contract<0,1>()));

    Tensor<float,4> c;
    c[0] = 2;
    Tensor<float,4,Contravariant> d({3,5,7,11});
    auto s = c * d;

    ASSERT_EQ(s, (Tensor<float,4,Contravariant>({6,10,14,22})));

}

TEST(TensorTest, Indexing) {
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,0,0,0})),0);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({1,0,0,0})),1);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({2,0,0,0})),2);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({3,0,0,0})),3);

    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,1,0,0})),4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({1,1,0,0})),5);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({2,1,0,0})),6);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({3,1,0,0})),7);

    ASSERT_EQ((TensorHelper<size_t,4,4>::index({1,2,0,0})),9);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({1,3,0,0})),13);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({1,4,0,0})),17);


    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,4,0,1})),4*4*4*1 + 4*4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,3,0,2})),4*4*4*2 + 3*4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,2,0,3})),4*4*4*3 + 2*4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,1,0,4})),4*4*4*4 + 1*4);

    ASSERT_EQ((TensorHelper<size_t,4,4>::index(std::make_tuple(0,4,0,1))), 4*4*4*1 + 4*4);




    #if 0
    // this should be a compiler error because there are 5 parameters!
    ASSERT_EQ((TensorHelper<size_t,4,4>::index({0,0,0,0,0})),0);
    #endif
}


TEST(TensorTest, Indexing2) {
    {
        auto coords = TensorHelper<size_t,4,4>::dimension(0);
        ASSERT_EQ(coords, (tuple<size_t,size_t,size_t,size_t>(0,0,0,0)));
    }
    {
        auto coords = TensorHelper<size_t,4,4>::dimension(1);
        ASSERT_EQ(coords, (tuple<size_t,size_t,size_t,size_t>(1,0,0,0)));
    }
    {
        auto coords = TensorHelper<size_t,4,4>::dimension(4*4*4*3 + 4*2);
        ASSERT_EQ(coords, (tuple<size_t,size_t,size_t,size_t>(0,2,0,3)));
    }
}

TEST(GLBlockTest, Contract) {
    typedef Tensor<float, 2, Covariant, Contravariant> t01;
    typedef Tensor<float, 2, Contravariant, Covariant> t10;
    typedef Tensor<float, 4, Covariant, Contravariant> T01;
    typedef Tensor<float, 4, Contravariant, Covariant> T10;

    std::cout << "size: " << TensorSize<2, Covariant, Contravariant>::value << std::endl;

    t01 t({ 1,2, 3,4 });

    auto c = t.contract<1,0>();
    ASSERT_EQ(c({}), 5);
    c = t.contract<0,1>();
    ASSERT_EQ(c({}), 5);

    t10 u({2,3,4,63}); 
    auto d = u.contract<0,1>();
    ASSERT_EQ(d({}), 65);

    //      _          _            __              __
    T01 T({ 1,2,3,4, 5,6,7,8,  9,10,11,12, 13,14,15,16 });
    //      _          _            __              __
    T10 U({ 2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,63 });

    auto e = T.contract<0,1>();
    auto f = U.contract<0,1>();

    ASSERT_EQ(T.size(), 16);
    ASSERT_EQ(U.size(), 16);

    ASSERT_EQ(e({}), 1+6+11+16);
    ASSERT_EQ(f({}), 2+7+12+63);
}

TEST(TensorTest, Types) {
    ASSERT_TRUE((is_contravariant<0, Contravariant>::value));
    ASSERT_FALSE((is_contravariant<0, Covariant>::value));
    ASSERT_TRUE((is_contravariant<1, Covariant, Contravariant>::value));
    ASSERT_TRUE((is_contravariant<1, Contravariant, Contravariant>::value));
    ASSERT_TRUE((is_contravariant<4, Covariant, Covariant, Covariant, Covariant, Contravariant>::value));
    ASSERT_FALSE((is_contravariant<3, Covariant, Covariant, Covariant, Covariant, Contravariant>::value));
    ASSERT_FALSE((is_contravariant<2, Covariant, Covariant, Covariant, Covariant, Contravariant>::value));
    ASSERT_FALSE((is_contravariant<1, Covariant, Covariant, Covariant, Covariant, Contravariant>::value));
    ASSERT_FALSE((is_contravariant<0, Covariant, Covariant, Covariant, Covariant, Contravariant>::value));

    ASSERT_TRUE((is_covariant<0, Covariant>::value));
    ASSERT_FALSE((is_covariant<0, Contravariant>::value));
    ASSERT_TRUE((is_covariant<1, Contravariant, Covariant>::value));
    ASSERT_TRUE((is_covariant<1, Covariant, Covariant>::value));
    ASSERT_TRUE((is_covariant<4, Contravariant, Contravariant, Contravariant, Contravariant, Covariant>::value));
    ASSERT_FALSE((is_covariant<3, Contravariant, Contravariant, Contravariant, Contravariant, Covariant>::value));
    ASSERT_FALSE((is_covariant<2, Contravariant, Contravariant, Contravariant, Contravariant, Covariant>::value));
    ASSERT_FALSE((is_covariant<1, Contravariant, Contravariant, Contravariant, Contravariant, Covariant>::value));
    ASSERT_FALSE((is_covariant<0, Contravariant, Contravariant, Contravariant, Contravariant, Covariant>::value));

    std::cerr << typeid(contraction_type<0,1,Contravariant,Covariant>::type).name() << std::endl;
    std::cerr << typeid(variance_container<>).name() << std::endl;

    ASSERT_TRUE((std::is_same<
        variance_container<>,
        typename contraction_type<0,1,Contravariant,Covariant>::type
    >::value));

    ASSERT_TRUE((std::is_same<
        variance_container<Contravariant>,
        typename contraction_type<0,1,Contravariant,Covariant,Contravariant>::type
    >::value));

    ASSERT_TRUE((std::is_same<
        variance_container<Contravariant>,
        typename contraction_type<1,2,Contravariant,Covariant,Contravariant>::type
    >::value));

    ASSERT_TRUE((std::is_same<
        variance_container<Covariant>,
        typename contraction_type<0,1,Contravariant,Covariant,Covariant>::type
    >::value));

    ASSERT_TRUE((std::is_same<
        variance_container<Covariant>,
        typename contraction_type<0,2,Contravariant,Covariant,Covariant>::type
    >::value));
}