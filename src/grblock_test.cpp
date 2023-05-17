#include "grblock.hpp"

#include <filesystem>
#include <iostream>
#include <functional>
#include <array>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>

// calculates the schwartzchild metric with schwarzschild radius of 1 in relativistic coords
GRElement Schwarzschild(float r, float theta) {
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);

    // construct the metric
    metric_type g; // initialize to zero
    g({0,0}) = 1./r - 1.; // time coord
    g({1,1}) = r / (r - 1.); // radial coord
    g({2,2}) = r * r; // theta
    g({3,3}) = r * r * sin_theta * sin_theta; // phi

    // inverse of a diagonal matrix is the inverse of the diagonals.
    metric_type inv;
    inv({0,0}) = r / (1. - r);
    inv({1,1}) = (r - 1.) / r;
    inv({2,2}) = 1. / (r * r);
    inv({3,3}) = 1. / (r * r * sin_theta * sin_theta);

    // calculate the partial derivatives
    metric_derivative_type dg;
    dg({1,0,0}) = -1/(r * r); // radial derivative of time coord
    dg({1,1,1}) = -r / (r - 1) / (r - 1) + 1 / (r - 1); // radial derivative of radial coord
    dg({1,2,2}) = 2. * r; // radial derivative of theta
    dg({1,3,3}) = 2. * r * sin_theta * sin_theta; // radial derivative of phi

    dg({2,3,3}) = r * r * 2. * sin_theta * cos_theta; // theta derivative of phi

    // calcluate the partial derivatives of the inverse
    inverse_derivative_type dinv;
    dinv({0,0,0}) = (r + 1.) / (1. - r) / (1. - r);
    dinv({0,1,1}) = (r - 1) / r / r;
    dinv({0,2,2}) = -2. / (r * r * r);
    dinv({0,3,3}) = -2. * r *     sin_theta * sin_theta / (r * r * sin_theta * sin_theta) / (r * r * sin_theta * sin_theta);
    dinv({1,3,3}) = -4. * r * r * sin_theta * cos_theta / (r * r * sin_theta * sin_theta) / (r * r * sin_theta * sin_theta);


    // calculate the second derivative
    metric_2nd_derivative_type d2g;
    d2g({1,1,0,0}) = 2. / (r * r * r); // radial second derivative of time coord
    d2g({1,1,1,1}) = 2. * r / (r - 1.) / (r - 1.) / (r - 1.) - 2. / (r - 1.) / (r - 1.); // radial second derivative of radial coord
    d2g({1,1,2,2}) = 2.; // radial second derivative of theta
    d2g({1,1,3,3}) = 2. * sin_theta * sin_theta; // radial second derivative of phi

    d2g({2,1,3,3}) = 4. * r * sin_theta * cos_theta; // theta, radial second derivative of theta
    d2g({2,2,3,3}) = 2. * r * r * ( cos_theta * cos_theta - sin_theta * sin_theta ); // theta second derivative of phi

    return GRElement{g,inv,dg,dinv,d2g};
}

TEST(GRBlockTest, Schwarzschild) {
    // define the Schwarzschild metric
    auto schwarzschild = Schwarzschild(2., 2.);

    // get the ricci tensor
    auto ricci = schwarzschild.ricci();

    ricci.print(std::cout);
    std::cout << std::endl;

    ASSERT_EQ(ricci({0,0}), 0.);

}