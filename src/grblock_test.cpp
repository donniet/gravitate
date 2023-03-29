#include "grblock.hpp"

#include <filesystem>
#include <iostream>
#include <functional>
#include <array>
#include <random>

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