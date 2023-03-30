
#include "grblock.hpp"

#include <iostream>
#include <algorithm>
#include <execution>
#include <vector>
#include <random>
#include <memory>
#include <string>
#include <array>
#include <map>
#include <queue>
#include <fstream>
#include <thread>
#include <deque>
#include <filesystem>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* creating to fix deprecated function in boost::compute */
#ifndef __GNUC__
namespace std {
    template<class RandomIt> 
    void random_shuffle(RandomIt first, RandomIt last) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(first, last, g);
    }
}
#endif

#include <boost/compute.hpp>

namespace compute = boost::compute;

using compute::_1;

using std::ostream;
using std::istream;
using std::cout;
using std::cerr;
using std::endl;
using std::cin;
using std::flush;
using std::string;

template<typename Number>
bool invertMatrix(const Number m[16], Number invOut[16])
{
    Number inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

#define BLOCK_SIZE 128

template<typename T>
ostream & raw_write(ostream & os, T const & obj) {
    return os.write(reinterpret_cast<const char *>(&obj), sizeof(T));
}

template<typename T>
istream & raw_read(istream & is, T & obj) {
    return is.read(reinterpret_cast<char *>(&obj), sizeof(T));
}

class Metric {
private:
    float g[16];          // metric tensor

    Metric(std::array<float, 16> const & m) : g{m[0], m[1], m[2], m[3],
                                                m[4], m[5], m[6], m[7],
                                                m[8], m[9], m[10], m[11],
                                                m[12], m[13], m[14], m[15]}
    { }

    Metric(float m[16]) : g{                    m[0], m[1], m[2], m[3],
                                                m[4], m[5], m[6], m[7],
                                                m[8], m[9], m[10], m[11],
                                                m[12], m[13], m[14], m[15]}
    { }
public:
    Metric() : g{ 0,0,0,0, 
                  0,0,0,0, 
                  0,0,0,0, 
                  0,0,0,0 } 
    { }

    float operator()(int i, int j) const {
        return g[i + j*4];
    }
    bool inv(Metric & out) const {
        return invertMatrix(g, out.g);
    }

    static Metric identity() {
        static float ident[16] = {          1,0,0,0, 
                                            0,1,0,0, 
                                            0,0,1,0, 
                                            0,0,0,1 };
        return Metric(ident);
    }
};


struct metric_op {
    float work[64];
    unsigned char dex[64][2];

    metric_op(Metric const & m, Contravariant const & v) {
        
        // std::transform(std::execution::par, work, work+64, work, )
    }
};



struct Curvature {
    float r[16];
};

/*
\begin{equation}
\begin{aligned}
R_{\mu \nu}=&\frac{1}{2}\, {\partial}_{\rho}{{g}^{\rho \sigma}}\,  {\partial}_{\nu}{{g}_{\mu \sigma}}\,  + \frac{1}{2}\, {\partial}_{\rho}{{g}^{\rho \sigma}}\,  {\partial}_{\mu}{{g}_{\nu \sigma}}\,  - \frac{1}{2}\, {\partial}_{\rho}{{g}^{\rho \sigma}}\,  {\partial}_{\sigma}{{g}_{\mu \nu}}\,  + \frac{1}{2}\, {g}^{\rho \sigma} {\partial}_{\nu \rho}{{g}_{\mu \sigma}}\, 
 + \frac{1}{2}\, {g}^{\rho \sigma} {\partial}_{\mu \rho}{{g}_{\nu \sigma}}\,\\
&  - \frac{1}{2}\, {g}^{\rho \sigma} {\partial}_{\rho \sigma}{{g}_{\mu \nu}}\,  - \frac{1}{2}\, {\partial}_{\nu}{{g}^{\rho \sigma}}\,  {\partial}_{\mu}{{g}_{\rho \sigma}}\,  - \frac{1}{2}\, {g}^{\rho \sigma} {\partial}_{\mu \nu}{{g}_{\rho \sigma}}\,  + \frac{1}{4}\, {g}^{\kappa \lambda} {\partial}_{\nu}{{g}_{\mu \kappa}}\,  {g}^{\rho \sigma} {\partial}_{\lambda}{{g}_{\rho \sigma}}\,  + \frac{1}{4}\, {g}^{\kappa \lambda} {\partial}_{\mu}{{g}_{\nu \kappa}}\,  {g}^{\rho \sigma} {\partial}_{\lambda}{{g}_{\rho \sigma}}\, \\
& - \frac{1}{4}\, {g}^{\kappa \lambda} {\partial}_{\kappa}{{g}_{\mu \nu}}\,  {g}^{\rho \sigma} {\partial}_{\lambda}{{g}_{\rho \sigma}}\,  - \frac{1}{4}\, {g}^{\kappa \lambda} {\partial}_{\mu}{{g}_{\kappa \rho}}\,  {g}^{\rho \sigma} {\partial}_{\nu}{{g}_{\lambda \sigma}}\,  - \frac{1}{2}\, {g}^{\kappa \lambda} {\partial}_{\kappa}{{g}_{\mu \rho}}\,  {g}^{\rho \sigma} {\partial}_{\sigma}{{g}_{\nu \lambda}}\,  + \frac{1}{2}\, {g}^{\kappa \lambda} {\partial}_{\kappa}{{g}_{\mu \rho}}\,  {g}^{\rho \sigma} {\partial}_{\lambda}{{g}_{\nu \sigma}} 
\end{aligned}
\end{equation}
*/
Curvature ricci(Metric const & m) {
    Curvature c;

    for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
        // c.r[i][j] = 0.5*sum(m.gd[])
    }}

    return c;
}

struct Block {
    Metric m[BLOCK_SIZE];
};

struct BlockIndex {
    int dex_[4];

    BlockIndex(int t, int x, int y, int z) : dex_{t, x, y, z} {}
};

class BlockRef {
private:
    string path_;
    BlockIndex dex_;

public:

};

class Universe {
private:
    string path_;

    // Metric (*initial_conidition)(Point p);

protected:
    bool loadBlock(int t, int x, int y, int z, Block & out) {
        return true;
    }
    bool saveBlock(int t, int x, int y, int z, Block const & in) {
        return true;
    }


public:

};


const char * kernels = BOOST_COMPUTE_STRINGIZE_SOURCE(


union matrix {
    float s[16];
    float16 v;
};

union coords {
    long s[4];
    long4 v;
};


size_t dex(long3 pos, long3 dim);
size_t dex2(long2 pos, long2 dim);
int invertMatrix(global union matrix * m, global union matrix * inv);


size_t dex(long3 pos, long3 dim) {
    return pos.x + dim.x * (pos.y + dim.y * pos.z);
}
size_t dex2(long2 pos, long2 dim) {
    return pos.x + dim.x * pos.y;
}

// kernel void ricci(
//     global float16 * metric_t0,
//     global float16 * metric_t1,
//     global float16 * metric_out,
//     float dt,
//     global float16 * ricci,
//     global float * ricci_scalar
// ) {
//     ulong3 pos;
//     ulong3 dim;
//     size_t pos.x = get_global_id(0),
//            dim.x = get_global_size(0),
//            pos.y = get_global_id(1),
//            dim.y = get_global_size(1),
//            pos.z = get_global_id(2),
//            dim.z = get_global_size(2);

//     float metric_d[16][4];

//     size_t m = dex(pos, dim);
    
//     for(int j = 0; j < 4; j++) {
//         for(int i = 0; i < 16; i++) {
//             if(j == 0) {
//                 metric_d[i][j] = (metric_t1[m] - metric_t0[m])/dt
//             } else if(pos[j-1] == 0 || pos[j-1] == dim[j-1]-1) {
//                 metric_d[i][j] = 0;
//             } else {
//                 metric_d[i][j] = (metric__t1[])
//             }
//         }
//     }
// }

// kernel void ricci_curvature(
//     global float16 * ricci,
//     global float16 * metric,
//     global float16 * metric_d,
//     global float16 * metric_dd
// ) {
//     size_t id = get_global_id(0);

// }

#define BORDER_X0 0
#define BORDER_X1 4
#define BORDER_Y0 1
#define BORDER_Y1 5
#define BORDER_Z0 2
#define BORDER_Z1 6
#define BORDER_T0 3
#define BORDER_T1 7

constant const int DERIV_X = 0;
constant const int DERIV_Y = 1;
constant const int DERIV_Z = 2;
constant const int DERIV_T = 3;

kernel void metricDerivativeKernel(
    global float16 * metric, // size N
    global float16 * metric_x0, // size N
    global float16 * metric_y0, // size N
    global float16 * metric_z0, // size N
    global float16 * metric_t0, // size N
    global float16 * deriv // size 4*N
) {
    long4 pos;
    union coords n;
    long3 dim;
    pos.x = get_global_id(0);
    dim.x = get_global_size(0);
    pos.y = get_global_id(1);
    dim.y = get_global_size(1);
    pos.z = get_global_id(2);
    dim.z = get_global_size(2);
    pos.w = 0;

    size_t i = dex(pos.xyz, dim);
    for(int k = 0; k < 4; k++) {
        n.v = pos;
        n.s[k]--;

        float16 p0;
        if(n.s[k] >= 0) {
            p0 = metric[dex(pos.xyz, dim)];
        } else {
            switch(k) {
            case DERIV_X: p0 = metric_x0[dex(long3(dim.x-1, pos.y, pos.z), dim.xyz)]; break;
            case DERIV_Y: p0 = metric_y0[dex(long3(pos.x, dim.y-1, pos.z), dim.xyz)]; break;
            case DERIV_Z: p0 = metric_z0[dex(long3(pos.x, pos.y, dim.z-1), dim.xyz)]; break;
            case DERIV_T: p0 = metric_t0[dex(pos.xyz, dim.xyz)]; break;
            }
        }

        deriv[i+k] = metric[i] - p0;
    }
}


kernel void invertMatrixKernel(
    global float16 * input,
    global float16 * inverse,
    global int * success
) {
    int i = get_global_id(0);

    success[i] = invertMatrix((global union matrix*)&input[i], (global union matrix*)&inverse[i]);
}


int invertMatrix(
    global union matrix * m,
    global union matrix * inv
) {
    float det;
    int i;

    inv->s[0] = m->s[5]  * m->s[10] * m->s[15] - 
             m->s[5]  * m->s[11] * m->s[14] - 
             m->s[9]  * m->s[6]  * m->s[15] + 
             m->s[9]  * m->s[7]  * m->s[14] +
             m->s[13] * m->s[6]  * m->s[11] - 
             m->s[13] * m->s[7]  * m->s[10];

    inv->s[4] = -m->s[4]  * m->s[10] * m->s[15] + 
              m->s[4]  * m->s[11] * m->s[14] + 
              m->s[8]  * m->s[6]  * m->s[15] - 
              m->s[8]  * m->s[7]  * m->s[14] - 
              m->s[12] * m->s[6]  * m->s[11] + 
              m->s[12] * m->s[7]  * m->s[10];

    inv->s[8] = m->s[4]  * m->s[9] * m->s[15] - 
             m->s[4]  * m->s[11] * m->s[13] - 
             m->s[8]  * m->s[5] * m->s[15] + 
             m->s[8]  * m->s[7] * m->s[13] + 
             m->s[12] * m->s[5] * m->s[11] - 
             m->s[12] * m->s[7] * m->s[9];

    inv->s[12] = -m->s[4]  * m->s[9] * m->s[14] + 
               m->s[4]  * m->s[10] * m->s[13] +
               m->s[8]  * m->s[5] * m->s[14] - 
               m->s[8]  * m->s[6] * m->s[13] - 
               m->s[12] * m->s[5] * m->s[10] + 
               m->s[12] * m->s[6] * m->s[9];

    inv->s[1] = -m->s[1]  * m->s[10] * m->s[15] + 
              m->s[1]  * m->s[11] * m->s[14] + 
              m->s[9]  * m->s[2] * m->s[15] - 
              m->s[9]  * m->s[3] * m->s[14] - 
              m->s[13] * m->s[2] * m->s[11] + 
              m->s[13] * m->s[3] * m->s[10];

    inv->s[5] = m->s[0]  * m->s[10] * m->s[15] - 
             m->s[0]  * m->s[11] * m->s[14] - 
             m->s[8]  * m->s[2] * m->s[15] + 
             m->s[8]  * m->s[3] * m->s[14] + 
             m->s[12] * m->s[2] * m->s[11] - 
             m->s[12] * m->s[3] * m->s[10];

    inv->s[9] = -m->s[0]  * m->s[9] * m->s[15] + 
              m->s[0]  * m->s[11] * m->s[13] + 
              m->s[8]  * m->s[1] * m->s[15] - 
              m->s[8]  * m->s[3] * m->s[13] - 
              m->s[12] * m->s[1] * m->s[11] + 
              m->s[12] * m->s[3] * m->s[9];

    inv->s[13] = m->s[0]  * m->s[9] * m->s[14] - 
              m->s[0]  * m->s[10] * m->s[13] - 
              m->s[8]  * m->s[1] * m->s[14] + 
              m->s[8]  * m->s[2] * m->s[13] + 
              m->s[12] * m->s[1] * m->s[10] - 
              m->s[12] * m->s[2] * m->s[9];

    inv->s[2] = m->s[1]  * m->s[6] * m->s[15] - 
             m->s[1]  * m->s[7] * m->s[14] - 
             m->s[5]  * m->s[2] * m->s[15] + 
             m->s[5]  * m->s[3] * m->s[14] + 
             m->s[13] * m->s[2] * m->s[7] - 
             m->s[13] * m->s[3] * m->s[6];

    inv->s[6] = -m->s[0]  * m->s[6] * m->s[15] + 
              m->s[0]  * m->s[7] * m->s[14] + 
              m->s[4]  * m->s[2] * m->s[15] - 
              m->s[4]  * m->s[3] * m->s[14] - 
              m->s[12] * m->s[2] * m->s[7] + 
              m->s[12] * m->s[3] * m->s[6];

    inv->s[10] = m->s[0]  * m->s[5] * m->s[15] - 
              m->s[0]  * m->s[7] * m->s[13] - 
              m->s[4]  * m->s[1] * m->s[15] + 
              m->s[4]  * m->s[3] * m->s[13] + 
              m->s[12] * m->s[1] * m->s[7] - 
              m->s[12] * m->s[3] * m->s[5];

    inv->s[14] = -m->s[0]  * m->s[5] * m->s[14] + 
               m->s[0]  * m->s[6] * m->s[13] + 
               m->s[4]  * m->s[1] * m->s[14] - 
               m->s[4]  * m->s[2] * m->s[13] - 
               m->s[12] * m->s[1] * m->s[6] + 
               m->s[12] * m->s[2] * m->s[5];

    inv->s[3] = -m->s[1] * m->s[6] * m->s[11] + 
              m->s[1] * m->s[7] * m->s[10] + 
              m->s[5] * m->s[2] * m->s[11] - 
              m->s[5] * m->s[3] * m->s[10] - 
              m->s[9] * m->s[2] * m->s[7] + 
              m->s[9] * m->s[3] * m->s[6];

    inv->s[7] = m->s[0] * m->s[6] * m->s[11] - 
             m->s[0] * m->s[7] * m->s[10] - 
             m->s[4] * m->s[2] * m->s[11] + 
             m->s[4] * m->s[3] * m->s[10] + 
             m->s[8] * m->s[2] * m->s[7] - 
             m->s[8] * m->s[3] * m->s[6];

    inv->s[11] = -m->s[0] * m->s[5] * m->s[11] + 
               m->s[0] * m->s[7] * m->s[9] + 
               m->s[4] * m->s[1] * m->s[11] - 
               m->s[4] * m->s[3] * m->s[9] - 
               m->s[8] * m->s[1] * m->s[7] + 
               m->s[8] * m->s[3] * m->s[5];

    inv->s[15] = m->s[0] * m->s[5] * m->s[10] - 
              m->s[0] * m->s[6] * m->s[9] - 
              m->s[4] * m->s[1] * m->s[10] + 
              m->s[4] * m->s[2] * m->s[9] + 
              m->s[8] * m->s[1] * m->s[6] - 
              m->s[8] * m->s[2] * m->s[5];

    det = m->s[0] * inv->s[0] + m->s[1] * inv->s[4] + m->s[2] * inv->s[8] + m->s[3] * inv->s[12];

    if (det == 0)
        return 0;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        inv->s[i] = inv->s[i] * det;

    return 1;
}

);

BOOST_COMPUTE_FUNCTION(compute::float16_, minikowski, (), {
    float16 m;
    
    m.s0123 = (float4)(2,  0,  0,  0);
    m.s4567 = (float4)(0, -1,  0,  0);
    m.s89AB = (float4)(0,  0, -1,  0);
    m.sCDEF = (float4)(0,  0,  0, -1);

    return m;
});

struct deriv {
    compute::program prog;
    size_t block_size;

    void operator()(
        compute::mapped_view<compute::float16_> & m,
        compute::mapped_view<compute::float16_> & x0,
        compute::mapped_view<compute::float16_> & y0,
        compute::mapped_view<compute::float16_> & z0,
        compute::mapped_view<compute::float16_> & t0,
        compute::mapped_view<compute::float16_> & d,
        compute::command_queue & queue
    ) {
        cout << "derivative..." << flush;
        auto derivKernel = prog.create_kernel("metricDerivativeKernel");
        derivKernel.set_args(
            m.get_buffer(),
            x0.get_buffer(),
            y0.get_buffer(),
            z0.get_buffer(),
            t0.get_buffer(),
            d.get_buffer()
        );
        queue.enqueue_nd_range_kernel<3>(
            derivKernel, 
            compute::extents<3>(0),
            compute::extents<3>(block_size),
            compute::extents<3>(1));
        cout << "done." << endl;
    }
};

struct invert {
    compute::program prog;

    void operator()(
        compute::mapped_view<compute::float16_> & m,
        compute::mapped_view<compute::float16_> & inv,
        compute::mapped_view<compute::int_> & succ,
        compute::command_queue & queue
    ) {
        
        cout << "inverting metric..." << flush;
        auto invertMatrixKernel = prog.create_kernel("invertMatrixKernel");
        invertMatrixKernel.set_args(
            m.get_buffer(), 
            inv.get_buffer(), 
            succ.get_buffer());
        queue.enqueue_1d_range_kernel(invertMatrixKernel, 0, m.size(), queue.get_device().max_work_group_size());
        cout << "done." << endl;
    }
};



int main(int ac, char * av[]) {
    compute::device gpu = compute::system::default_device();

    compute::context ctx(gpu);
    compute::command_queue queue(ctx, gpu);

    cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << CL_DEVICE_MAX_MEM_ALLOC_SIZE << endl;

    std::size_t size = 64;

    cout << "compiling kernels... " << flush;
    auto prog = compute::program::create_with_source(kernels, ctx);
    try {
        prog.build();
    } 
    catch(boost::exception & e) {
        cerr << endl << prog.get_build_info<string>(CL_PROGRAM_BUILD_LOG, gpu) << endl;
        return -1;
    }
    cout << "done." << endl;


    cout << "allocating memory... " << flush;
    auto metric = std::vector<compute::float16_>(size * size * size);
    auto inverse = std::vector<compute::float16_>(size * size * size);
    auto success = std::vector<compute::int_>(size * size * size);
    auto metric_d = std::vector<compute::float16_>(size * size * size * 4);
    auto metric_dd = std::vector<compute::float16_>(size * size * size * 4 * 2);
    auto ricci = std::vector<compute::float16_>(size * size * size);
    auto curve = std::vector<compute::float_>(size * size * size);
    cout << "done." << endl;

    cout << "metric size: " << metric.size() << endl;
    cout << "metric_d size: " << metric_d.size() << endl;
    cout << "metric_dd size: " << metric_dd.size() << endl;
    cout << "ricci size: " << ricci.size() << endl;
    cout << "curve size: " << curve.size() << endl;

    cout << "mapping memory... " << flush;
    compute::mapped_view<decltype(metric)::value_type> metric_map(metric.data(), metric.size(), ctx);
    compute::mapped_view<decltype(inverse)::value_type> inverse_map(inverse.data(), inverse.size(), ctx);
    compute::mapped_view<decltype(success)::value_type> success_map(success.data(), success.size(), ctx);
    compute::mapped_view<decltype(metric_d)::value_type> metric_d_map(metric_d.data(), metric_d.size(), ctx);
    compute::mapped_view<decltype(metric_dd)::value_type> metric_dd_map(metric_dd.data(), metric_dd.size(), ctx);
    compute::mapped_view<decltype(ricci)::value_type> ricci_map(ricci.data(), ricci.size(), ctx);
    compute::mapped_view<decltype(curve)::value_type> curve_map(curve.data(), curve.size(), ctx);
    cout << "done." << endl;

    cout << "initializing to minikowski metric..." << flush;
    compute::generate(metric_map.begin(), metric_map.end(), minikowski, queue);
    cout << "done." << endl;

    invert{prog}(metric_map, inverse_map, success_map, queue);
    

    queue.finish();

    for(int i = 0; i < 10; i++) {
        cout << "m: " << metric[i] << ", inv: " << inverse[i] << endl;
    }




    


    int a;
    cout << "pausing... " << flush;
    std::cin >> a;
    cout << "done." << endl;


    return 0;
}