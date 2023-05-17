#pragma once

#include <cstddef>
#include <tuple>
#include <cmath>

#include "tensor.hpp"

using std::tie;

typedef Tensor<float,4,Contravariant>                               vector_type;
typedef Tensor<float,4,Covariant,Covariant>                         metric_type;
typedef Tensor<float,4,Covariant,Covariant,Covariant>               metric_derivative_type;
typedef Tensor<float,4,Covariant,Contravariant,Contravariant>       inverse_derivative_type;
typedef Tensor<float,4,Covariant,Covariant,Covariant,Covariant>     metric_2nd_derivative_type;
typedef Tensor<float,4,Contravariant, Covariant,Covariant>          connection_type;
typedef Tensor<float,4,Covariant,Covariant>                         ricci_type;
typedef Tensor<float,4,Contravariant,Covariant,Covariant,Covariant> riemann_type;




struct GRElement {
    // vector_type corner_min, corner_max;
    metric_type metric;
    metric_type inverse;
    metric_derivative_type metric_derivative;
    inverse_derivative_type inverse_derivative;
    metric_2nd_derivative_type metric_2nd_derivative;


    ricci_type ricci() const;
    ricci_type ricci2() const;
    connection_type connection() const;
    riemann_type riemann() const;
};

/* R_{a b} =  1/2 \partial_{c}(g^{c d}) \partial_{a}(g_{b d}) +  
              1/2 \partial_{c}(g^{c d}) \partial_{b}(g_{a d}) -  
              1/2 \partial_{c}(g^{c d}) \partial_{d}(g_{a b}) -  
              1/2 \partial_{b}(g^{c d}) \partial_{a}(g_{c d}) + 
              g^{c d} ( 1/2 \partial_{a c}(g_{b d}) + 
                        1/2 \partial_{b c}(g_{a d}) -  
                        1/2 \partial_{c d}(g_{a b}) -  
                        1/2 \partial_{a b}(g_{c d}) +  
                        1/4 \partial_{a}(g_{b c}) g^{e f} \partial_{d}(g_{e f})
                       ) + 
              g^{c d} g^{e f} ( 1/4 \partial_{b}(g_{a c}) \partial_{d}(g_{e f}) -  
                                1/4 \partial_{c}(g_{a b}) \partial_{d}(g_{e f}) -  
                                1/2 \partial_{c}(g_{a e}) \partial_{f}(g_{b d}) +  
                                1/2 \partial_{c}(g_{a e}) \partial_{d}(g_{b f}) -  
                                1/4 \partial_{a}(g_{c e}) \partial_{b}(g_{d f})
                              )
*/
ricci_type GRElement::ricci() const {
    ricci_type ret;

    typedef Tensor<float,4,Covariant,Covariant,Covariant,Covariant,Covariant,Covariant> parenthetical_type;
    auto rbeg = ret.begin();

    std::for_each(/*std::execution::par_unseq, */rbeg, ret.end(), [&](float & v) {
        size_t u = &v - rbeg; // get the index to this element

        // get the indices of this element in the ricci tensor
        auto dims = ret.dimension(u);
        size_t a, b;
        tie(a, b) = dims;

        float sum = 0.;

        inverse_derivative.print(std::cout);
        std::cout << std::endl;
        metric_derivative.print(std::cout);
        std::cout << std::endl;

        // contractions on the first four terms

        for(size_t c = 0; c < parenthetical_type::dimensions; c++)
        for(size_t d = 0; d < parenthetical_type::dimensions; d++) {
            sum += inverse_derivative({c, c, d}) * metric_derivative({a, b, d}) 
                 + inverse_derivative({c, c, d}) * metric_derivative({b, a, d})
                 - inverse_derivative({c, c, d}) * metric_derivative({d, a, b})
                 - inverse_derivative({b, c, d}) * metric_derivative({a, c, d});

            if(std::isnan(sum)) {
                std::cerr << "sum is NaN after step " << c << ", " << d << std::endl;
                goto firstFourOuterLoop;
            }
        }
firstFourOuterLoop:
        sum *= 0.5;

        if(std::isnan(sum)) {
            std::cerr << "sum is NaN after first four terms" << std::endl;
            throw std::logic_error("sum is NaN after first four terms");
        }

        // the next four terms are evaluated similarly
        for(size_t c = 0; c < parenthetical_type::dimensions; c++)
        for(size_t d = 0; d < parenthetical_type::dimensions; d++) {
            float temp = metric_2nd_derivative({a, c, b, d})
                       + metric_2nd_derivative({b, c, a, d})
                       - metric_2nd_derivative({c, d, a, b})
                       - metric_2nd_derivative({a, b, c, d});
            temp *= 0.5 * inverse({c, d});

            sum += temp;
        }


        if(std::isnan(sum)) {
            std::cerr << "sum is NaN after second four terms" << std::endl;
            throw std::logic_error("sum is NaN after second four terms");
        }

        // the remaining six terms require four contractions
        for(size_t c = 0; c < parenthetical_type::dimensions; c++)
        for(size_t d = 0; d < parenthetical_type::dimensions; d++)
        for(size_t e = 0; e < parenthetical_type::dimensions; e++)
        for(size_t f = 0; f < parenthetical_type::dimensions; f++) {
            float temp = 0.25 * metric_derivative({a, b, c}) * metric_derivative({d, e, f})
                       + 0.25 * metric_derivative({b, a, c}) * metric_derivative({d, e, f})
                       - 0.25 * metric_derivative({c, a, b}) * metric_derivative({d, e, f})
                       - 0.5  * metric_derivative({c, a, e}) * metric_derivative({f, b, d})
                       + 0.5  * metric_derivative({c, a, e}) * metric_derivative({d, b, f})
                       - 0.25 * metric_derivative({a, c, e}) * metric_derivative({b, d, f});
            
            temp *= inverse({c, d}) * inverse({e, f});

            sum += temp;
        }

        if(std::isnan(sum)) {
            std::cerr << "sum is NaN after last six terms" << std::endl;
            throw std::logic_error("sum is NaN after last six terms");
        }

        v = sum;
    });

    

    return ret;
}

/*
calculates the ricci tensor from the connection and the connection derivatives
using this formula:
R_{ab} = \frac{1}{2} g^{cd} \( ∂_a ∂_c g_{bd} + ∂_b ∂_d g_{ac} - ∂_a ∂_d g_{bc} - ∂_b ∂_c g_{ad} \) +
         \frac{1}{2} \( g^{ce} Γ^d_{ec} - g^{de} Γ^c_{ed} \)\( Γ^e_{ad} - Γ^e_{ab} \)
*/
ricci_type GRElement::ricci2() const {
    // TODO: figure out a caching strategy for the metric inverse
    auto inv = invert(metric);
    auto conn = connection();

    ricci_type ret;

    typedef Tensor<float,4,Covariant,Covariant,Contravariant,Contravariant,Covariant,Covariant> parenthetical_type;

    auto rbeg = ret.begin();
    std::transform(std::execution::par_unseq, rbeg, ret.end(), rbeg, [&](float & v) -> float {
        size_t u = &v - rbeg; // get the index to this element

        // calculate the indices in this tensor
        auto dims = ret.dimension(u);

        // uncontract the indices to the parenthetical type
        auto uncontracted = tuple_cat(dims, tuple<size_t,size_t,size_t,size_t>{0,0,0,0});
        
        // get the initial index into the parenthetical type
        auto uncontracted_index = parenthetical_type::index(uncontracted);

        // get the stride assuming two contractions on the 2,4 3,5 indices
        static constexpr size_t stride = tensor_stride<4,2,3,4,5>::value;

        // add up all the contracted elements
        // TODO: this should be paralized, but it's trickier because there is no storage array for these elements

        float sum = 0;
        for(size_t i = 0; i < power<4,4>::value; ++i, uncontracted_index += stride) {
            // turn these back into dimensions
            auto pdims = parenthetical_type::dimension(uncontracted_index);

            // name all the dimensions
            size_t a, b, c, d, e, f;
            tie(a,b,c,d,e,f) = pdims;

            // now grab the elements of the contractions and add them to the sum
            sum += inv({c,d}) * ( metric_2nd_derivative({a,c,b,d}) +
                                  metric_2nd_derivative({b,d,a,c}) -
                                  metric_2nd_derivative({a,d,b,c}) -
                                  metric_2nd_derivative({b,c,a,d}) );

            sum += inv({c,e}) * conn({d,e,c}) - inv({d,e}) * conn({c,e,d});
        }

        return 0.5 * sum;
    });


    // A will hold the inner first parenthetical
    Tensor<float,4,Covariant,Covariant,Covariant,Covariant> A(false);



    auto abeg = A.begin();
    std::transform(std::execution::par_unseq, abeg, A.end(), abeg, [&](float & v) -> float {
        size_t u = &v - abeg; // get the index to this element

        // calculate the indices in this tensor
        auto dims = A.dimension(u);

        // name each of the indices
        size_t a, b, c, d;
        tie(a,b,c,d) = dims;

        return 0.5 * (metric_2nd_derivative({a,c,b,d}) + 
                      metric_2nd_derivative({b,d,a,c}) - 
                      metric_2nd_derivative({a,d,b,c}) - 
                      metric_2nd_derivative({b,c,a,d}));
    });



    return ret;
}


/*
calculates the metric parameters from the metric and derivative
Γ^l_{jk} = \frac{1}{2} g^{lr} \( ∂_k g_{rj} + ∂_j g_{rk} - ∂_r g_{jk} \)
*/
connection_type GRElement::connection() const {
    // TODO: figure out a caching strategy for the metric inverse
    auto inv = invert(metric);
    
    Tensor<float,4,Covariant,Covariant,Covariant> christoffel(false); // uninitialized

    auto beg = christoffel.begin();

    // first calculate the inside of the parents
    std::transform(std::execution::par_unseq, beg, christoffel.end(), beg, [&](float v) -> float {
        size_t d = &v - beg; // get the index to this element

        // turn this into index into a set of indices
        auto p = christoffel.dimension(d);
        size_t k = get<0>(p), r = get<1>(p), j = get<2>(p);

        return 0.5 * (metric_derivative({k, r, j}) + metric_derivative({j, r, k}) - metric_derivative({r, j, k}));
    });

    // multiply with contraction with the metric inverse
    return inv.multiplyAndContract<1,3>(christoffel);
}


struct GRBlock {
    
};
