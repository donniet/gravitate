#include "tensor.hpp"

#include <filesystem>
#include <iostream>
#include <functional>
#include <array>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>

TEST(TensorTest, Mult) {
    Tensor<float,2,Covariant>     a({1,2});
    Tensor<float,2,Contravariant> b({3,5});

    auto res = a * b;
}

TEST(TensorTest, Indexing) {
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,0,0,0)),0);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(1,0,0,0)),1);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(2,0,0,0)),2);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(3,0,0,0)),3);

    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,1,0,0)),4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(1,1,0,0)),5);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(2,1,0,0)),6);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(3,1,0,0)),7);

    ASSERT_EQ((TensorHelper<size_t,4,4>::index(1,2,0,0)),9);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(1,3,0,0)),13);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(1,4,0,0)),17);


    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,4,0,1)),4*4*4*1 + 4*4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,3,0,2)),4*4*4*2 + 3*4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,2,0,3)),4*4*4*3 + 2*4);
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,1,0,4)),4*4*4*4 + 1*4);

    ASSERT_EQ((TensorHelper<size_t,4,4>::index(std::make_tuple(0,4,0,1))), 4*4*4*1 + 4*4);


    #if 0
    // this should be a compiler error because there are 5 parameters!
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,0,0,0,0)),0);
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

TEST(TensorTest, Tuples) {
    tuple<int, int, int> a(0,1,3);
    auto head = tuple_head<2>(a);

    ASSERT_EQ(get<0>(head), 0);
    ASSERT_EQ(get<1>(head), 1);

    auto tail = tuple_tail<2>(a);
    ASSERT_EQ(get<0>(tail), 3);

    auto full = tuple_splice<2>(a, 2);
    ASSERT_EQ(get<0>(full), 0);
    ASSERT_EQ(get<1>(full), 1);
    ASSERT_EQ(get<2>(full), 2);
    ASSERT_EQ(get<3>(full), 3);

    auto full2 = tuple_splice<0>(a, tuple_splice<2>(a, 2));
}

TEST(GLBlockTest, Contract) {
    typedef Tensor<float, 2, Covariant, Contravariant> t01;
    typedef Tensor<float, 2, Contravariant, Covariant> t10;
    typedef Tensor<float, 4, Covariant, Contravariant> T01;
    typedef Tensor<float, 4, Contravariant, Covariant> T10;

    std::cout << "size: " << TensorSize<2, Covariant, Contravariant>::value << std::endl;

    t01 t({ 1,2, 3,4 });

    auto c = t.contract<1,0>();
    ASSERT_EQ(c(0), 5);
    c = t.contract<0,1>();
    ASSERT_EQ(c(0), 5);

    t10 u({2,3,4,63}); 
    auto d = u.contract<0,1>();
    ASSERT_EQ(d(0), 65);

    //      _          _            __              __
    T01 T({ 1,2,3,4, 5,6,7,8,  9,10,11,12, 13,14,15,16 });
    //      _          _            __              __
    T10 U({ 2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,63 });

    auto e = T.contract<0,1>();
    auto f = U.contract<0,1>();

    ASSERT_EQ(T.size(), 16);
    ASSERT_EQ(U.size(), 16);

    ASSERT_EQ(e(0), 1+6+11+16);
    ASSERT_EQ(f(0), 2+7+12+63);
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