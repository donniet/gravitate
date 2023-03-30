#include "grblock.hpp"

#include <filesystem>
#include <iostream>
#include <functional>
#include <array>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>

TEST(GRBlockTest, Indexing) {
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

    #if 0
    // this should be a compiler error because there are 5 parameters!
    ASSERT_EQ((TensorHelper<size_t,4,4>::index(0,0,0,0,0)),0);
    #endif
}

TEST(GRBlockTest, Indexing2) {
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

TEST(GRBlockTest, Types) {
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