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

    ASSERT_EQ((TensorHelper<size_t,4,4>::index(std::make_tuple(0,4,0,1))), 4*4*4*1 + 4*4);


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

TEST(GRBlockTest, Tuples) {
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

TEST(GRBlockTest, Indexing3) {
    typedef Tensor<float, 4, Covariant, Contravariant, Covariant, Covariant> t0100;
    t0100 t;
    ASSERT_EQ(t.size(), 4*4*4*4);

    typedef typename ContractionHelper<float, 4, 0, 1, Covariant, Contravariant, Covariant, Covariant>::type contracted_type;
    contracted_type c;

    ASSERT_EQ(c.size(), 4*4);

    std::array<size_t,t0100::siz()> indices;

    // parallel iota
    auto beg = indices.begin();
    std::for_each(std::execution::par_unseq, beg, beg + t0100::siz(), [&beg](auto & element) {
        element = &element - beg;
    });

    // determine the contraction distance in the original tensor
    size_t stride = Contraction<t0100,0,1>::stride();

    auto un = Contraction<t0100,0,1>::uncontract_indices(make_tuple<size_t,size_t>(3,2));
    ASSERT_EQ(get<0>(un), 0);
    ASSERT_EQ(get<1>(un), 0);
    ASSERT_EQ(get<2>(un), 3);
    ASSERT_EQ(get<3>(un), 2);

    std::cout << "stride: " << stride << std::endl;
    std::cout << "power: " << ContractionHelper<float, 4, 0, 1, Covariant, Contravariant, Contravariant>::stride() << std::endl;

    std::transform(std::execution::par_unseq, c.begin(), c.end(), c.begin(), [&c, &t](auto & element) {
        // first get the index
        auto index = &element - c.begin();

        // transform the index into tensor indices
        auto dims = c.dimension(index);

        // get the tensor index of the uncontracted tensor
        // t0100::helper_type::index_type uncontracted;
        // uncontract<0,1>(uncontracted, dims);

        // get the index in the uncontracted
        // t.index(uncontracted);
        // auto undims = tuple_splice(dims)

        return element;
    });

    size_t dex = 0;
    for(auto & i : indices) {
        std::cout << i << " ";
        ASSERT_EQ(i, dex++);
    }
    std::cout << std::endl;

    ASSERT_TRUE(true);
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