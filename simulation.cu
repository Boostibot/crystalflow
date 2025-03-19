#include "cuda_util.cuh"
#include "cuda_reduction.cuh"
#include "cuda_for.cuh"
#include "cuda_random.cuh"

#include "simulation.h"

typedef Sim_Real Real;

#include <assert.h>
#ifndef ASSERT
    #define ASSERT(x)
#endif

thread_local static cudaEvent_t _cuda_timer_start = NULL;
thread_local static cudaEvent_t _cuda_timer_stop = NULL;

void cuda_timer_start()
{
    if(_cuda_timer_start == NULL || _cuda_timer_stop == NULL)
    {
        CUDA_TEST(cudaEventCreate(&_cuda_timer_start));
        CUDA_TEST(cudaEventCreate(&_cuda_timer_stop));
    }
    CUDA_TEST(cudaEventRecord(_cuda_timer_start, 0));
}

double cuda_timer_stop()
{
    CUDA_TEST(cudaEventRecord(_cuda_timer_stop, 0));
    CUDA_TEST(cudaEventSynchronize(_cuda_timer_stop));

    float time = 0;
    CUDA_TEST(cudaEventElapsedTime(&time, _cuda_timer_start, _cuda_timer_stop));
    return (double) time / 1000;
}

template <typename T>
void sim_modify_T(Real* device_memory, T* host_memory, size_t count, Sim_Modify modify)
{
    static T* static_device = NULL;
    static size_t static_size = 0;

    if(sizeof(Real) != sizeof(T))
    {
        if(static_size < count)
        {
            cuda_realloc_in_place((void**) &static_device, count*sizeof(T), static_size*sizeof(T), 0);
            static_size = count;
        }

        T* temp_device = static_device;
        if(modify == MODIFY_UPLOAD)
        {
            //Upload: host -> static -> device
            CUDA_DEBUG_TEST(cudaMemcpy(temp_device, host_memory, count*sizeof(T), cudaMemcpyHostToDevice));
            cuda_for(0, (int) count, [=]SHARED(int i){
                device_memory[i] = (Real) temp_device[i];
            });
        }
        else
        {
            //download: device -> static -> host
            cuda_for(0, (int) count, [=]SHARED(int i){
                temp_device[i] = (T) device_memory[i];
            });
            CUDA_DEBUG_TEST(cudaMemcpy(host_memory, temp_device, count*sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        if(modify == MODIFY_UPLOAD)
            CUDA_DEBUG_TEST(cudaMemcpy(device_memory, host_memory, count*sizeof(T), cudaMemcpyHostToDevice));
        else
            CUDA_DEBUG_TEST(cudaMemcpy(host_memory, device_memory, count*sizeof(T), cudaMemcpyDeviceToHost));
    }
}

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify)
{
    if(modify == MODIFY_UPLOAD)
        CUDA_DEBUG_TEST(cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice));
    else
        CUDA_DEBUG_TEST(cudaMemcpy(host_memory, device_memory, size, cudaMemcpyDeviceToHost));
}

extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t count, Sim_Modify modify)
{   
    sim_modify_T(device_memory, host_memory, count, modify);
}

extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t count, Sim_Modify modify)
{   
    sim_modify_T(device_memory, host_memory, count, modify);
}


extern "C" bool sim_mut_state_init(Sim_Mut_State* state, int32_t nx, int32_t ny)
{
    sim_mut_state_deinit(state);
    state->nx = nx;
    state->ny = ny;
    size_t bytes = (size_t) nx * (size_t) ny * sizeof(Real); 
    CUDA_TEST(cudaMalloc(&state->rho, bytes));
    CUDA_TEST(cudaMalloc(&state->ux, bytes));
    CUDA_TEST(cudaMalloc(&state->uy, bytes));
    return true;
}
extern "C" bool sim_const_state_init(Sim_Const_State* state, int32_t nx, int32_t ny)
{
    sim_const_state_deinit(state);
    state->nx = nx;
    state->ny = ny;
    size_t bytes = (size_t) nx * (size_t) ny * sizeof(Real); 

    CUDA_TEST(cudaMalloc(&state->flags, (size_t) nx * (size_t) ny * sizeof(Sim_Flags)));
    CUDA_TEST(cudaMalloc(&state->set_rho, bytes));
    CUDA_TEST(cudaMalloc(&state->set_ux, bytes));
    CUDA_TEST(cudaMalloc(&state->set_uy, bytes));

    CUDA_TEST(cudaMalloc(&state->set_dx_rho, bytes));
    CUDA_TEST(cudaMalloc(&state->set_dx_ux, bytes));
    CUDA_TEST(cudaMalloc(&state->set_dx_uy, bytes));

    CUDA_TEST(cudaMalloc(&state->set_dy_rho, bytes));
    CUDA_TEST(cudaMalloc(&state->set_dy_ux, bytes));
    CUDA_TEST(cudaMalloc(&state->set_dy_uy, bytes));
    return true;
}

extern "C" void sim_mut_state_deinit(Sim_Mut_State* state)
{
    CUDA_TEST(cudaFree(state->rho));
    CUDA_TEST(cudaFree(state->ux));
    CUDA_TEST(cudaFree(state->uy));
    memset(state, 0, sizeof *state);
}
extern "C" void sim_const_state_deinit(Sim_Const_State* state)
{
    CUDA_TEST(cudaFree(state->flags));

    CUDA_TEST(cudaFree(state->set_rho));
    CUDA_TEST(cudaFree(state->set_ux));
    CUDA_TEST(cudaFree(state->set_uy));

    CUDA_TEST(cudaFree(state->set_dx_rho));
    CUDA_TEST(cudaFree(state->set_dx_ux));
    CUDA_TEST(cudaFree(state->set_dx_uy));

    CUDA_TEST(cudaFree(state->set_dy_rho));
    CUDA_TEST(cudaFree(state->set_dy_ux));
    CUDA_TEST(cudaFree(state->set_dy_uy));
    memset(state, 0, sizeof *state);
}

struct Vars {
    Real rho;
    Real ux;
    Real uy;
};

extern "C" double sim_step(Sim_Mut_State* next, const Sim_Mut_State* prev, Sim_Const_State* const_state, Sim_Params params)
{
    csize nx = next->nx;
    csize ny = next->ny;

    Real* prev_rhos = prev->rho;
    Real* prev_uxs = prev->ux;
    Real* prev_uys = prev->uy;

    Real* next_rhos = next->rho;
    Real* next_uxs = next->ux;
    Real* next_uys = next->uy;

    Sim_Real* set_rho = const_state->set_rho;
    Sim_Real* set_ux = const_state->set_ux;
    Sim_Real* set_uy = const_state->set_uy;

    Sim_Real* set_dx_rho = const_state->set_dx_rho;
    Sim_Real* set_dx_ux = const_state->set_dx_ux;
    Sim_Real* set_dx_uy = const_state->set_dx_uy;

    Sim_Real* set_dy_rho = const_state->set_dy_rho;
    Sim_Real* set_dy_ux = const_state->set_dy_ux;
    Sim_Real* set_dy_uy = const_state->set_dy_uy;

    Sim_Flags* flags = const_state->flags;
    Real dt = params.dt;
    Real dx = params.region_width / nx;
    Real dy = params.region_height / ny;
    Real lambda = params.dynamic_viscosity;
    Real mu = params.second_viscosity;
    
    Real R_spec = 287.052874;
    Real T = params.temperature;
    //TODO: try the second version of tiled for as well - should be faster more kernels where most of the time
    // is spent computing.

    #define PRINT_F(x) printf(#x " = %e\n", (double) (x))

    #define PRINT_VARS(v) printf(#v " = {rho:%e ux:%e uy:%e}\n", (v).rho, (v).ux, (v).uy)

    //Precaclulate first order derivations
    cuda_tiled_for_2D<1, 1, Vars>(0, 0, nx, ny, 
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) {
                csize i = x + y*nx;
                Sim_Flags cflags = flags[i];
                
                return Vars{
                    cflags & SIM_SET_RHO ? set_rho[i] : prev_rhos[i],
                    cflags & SIM_SET_UX ? set_ux[i] : prev_uxs[i],
                    cflags & SIM_SET_UY ? set_uy[i] : prev_uys[i],
                };
            }
            else
                return Vars{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Vars* __restrict__ shared){
            csize i = x + y*nx;
            Sim_Flags cflags = flags[i];
            Vars vc = shared[tx + ty*tile_size_x];
            
            // next_rhos[i] = vc.rho;
            // next_uxs[i] = vc.ux;
            // next_uys[i] = vc.uy;

            // return;
            Vars vn = shared[tx + (ty+1)*tile_size_x];
            Vars vs = shared[tx + (ty-1)*tile_size_x];
            Vars ve = shared[(tx+1) + ty*tile_size_x];
            Vars vw = shared[(tx-1) + ty*tile_size_x];

            Vars vne = shared[(tx+1) + (ty+1)*tile_size_x];
            Vars vnw = shared[(tx-1) + (ty+1)*tile_size_x];
            Vars vse = shared[(tx+1) + (ty-1)*tile_size_x];
            Vars vsw = shared[(tx-1) + (ty-1)*tile_size_x];

            //TODO
            Real Fx = 0; 
            Real Fy = 0;

            Real rho = vc.rho;
            Real ux = vc.ux;
            Real uy = vc.uy;

            Real dx_rho = (ve.rho - vw.rho)/(2*dx);
            Real dx_ux = (ve.ux - vw.ux)/(2*dx);
            Real dx_uy = (ve.uy - vw.uy)/(2*dx);

            Real dy_rho = (vn.rho - vs.rho)/(2*dy);
            Real dy_ux = (vn.ux - vs.ux)/(2*dy);
            Real dy_uy = (vn.uy - vs.uy)/(2*dy);

            Real dxx_ux = (ve.ux + 2*vc.ux - vw.ux)/(dx*dx);
            Real dxx_uy = (ve.uy + 2*vc.uy - vw.uy)/(dx*dx);

            Real dyy_ux = (vn.ux + 2*vc.ux - vs.ux)/(dy*dy);
            Real dyy_uy = (vn.uy + 2*vc.uy - vs.uy)/(dy*dy);

            Real dxy_ux = (vne.ux - vnw.ux - vse.ux + vsw.ux)/(4*dx*dy);
            Real dxy_uy = (vne.uy - vnw.uy - vse.uy + vsw.uy)/(4*dx*dy);

            #if 0
            Real ux_dx_ux = ux >= 0 ? ux*(vc.ux - vw.ux)/dx : ux*(ve.ux - vc.ux)/dx;
            Real ux_dx_uy = ux >= 0 ? ux*(vc.uy - vw.uy)/dx : ux*(ve.uy - vc.uy)/dx;
            Real uy_dy_ux = uy >= 0 ? uy*(vc.ux - vs.ux)/dy : uy*(vn.ux - vc.ux)/dy;
            Real uy_dy_uy = uy >= 0 ? uy*(vc.uy - vs.uy)/dy : uy*(vn.uy - vc.uy)/dy;

            Real ux_dx_rho = ux >= 0 ? ux*(vc.rho - vw.rho)/dx : ux*(ve.rho - vc.rho)/dx;
            Real uy_dy_rho = uy >= 0 ? uy*(vc.rho - vs.rho)/dy : uy*(vn.rho - vc.rho)/dy;
            #else

            Real ux_dx_ux = true ? ux*(vc.ux - vw.ux)/dx : ux*(ve.ux - vc.ux)/dx;
            Real ux_dx_uy = true ? ux*(vc.uy - vw.uy)/dx : ux*(ve.uy - vc.uy)/dx;
            Real uy_dy_ux = true ? uy*(vc.ux - vs.ux)/dy : uy*(vn.ux - vc.ux)/dy;
            Real uy_dy_uy = true ? uy*(vc.uy - vs.uy)/dy : uy*(vn.uy - vc.uy)/dy;

            Real ux_dx_rho = true ? ux*(vc.rho - vw.rho)/dx : ux*(ve.rho - vc.rho)/dx;
            Real uy_dy_rho = true ? uy*(vc.rho - vs.rho)/dy : uy*(vn.rho - vc.rho)/dy;
            #endif

            if(cflags) {
                if(cflags & SIM_SET_DX_UX) dx_ux = set_dx_ux[i];
                if(cflags & SIM_SET_DX_UY) dx_uy = set_dx_uy[i];
                if(cflags & SIM_SET_DY_UX) dy_ux = set_dy_ux[i];
                if(cflags & SIM_SET_DY_UY) dy_uy = set_dy_uy[i];

                if(cflags & SIM_SET_DX_UX) ux_dx_ux = ux*dx_ux;
                if(cflags & SIM_SET_DX_UY) ux_dx_uy = ux*dx_uy;
                if(cflags & SIM_SET_DY_UX) uy_dy_ux = uy*dy_ux;
                if(cflags & SIM_SET_DY_UY) uy_dy_uy = uy*dy_uy;

                if(cflags & SIM_SET_DX_RHO) dx_rho = set_dx_rho[i];
                if(cflags & SIM_SET_DY_RHO) dy_rho = set_dy_rho[i];
                if(cflags & SIM_SET_DX_RHO) ux_dx_rho = ux*dx_rho;
                if(cflags & SIM_SET_DY_RHO) uy_dy_rho = uy*dy_rho;
            }

            #if 1
            Real dt_rho = -(ux_dx_rho + uy_dy_rho) + rho*(dx_ux + dy_uy);
            Real dt_ux = -(ux_dx_ux + uy_dy_ux) - R_spec*T/rho*dx_rho + Fx
                + 1/rho*((lambda + 2*mu)*dxx_ux + lambda*dxy_uy + mu*dyy_ux + mu*dxy_uy);
            Real dt_uy = -(ux_dx_uy + uy_dy_uy) - R_spec*T/rho*dy_rho + Fy
                + 1/rho*((lambda + 2*mu)*dyy_uy + lambda*dxy_ux + mu*dxx_uy + mu*dxy_ux);
            #else
            Real dt_rho = -(ux_dx_rho + uy_dy_rho) + rho*(dx_ux + dy_uy);
            Real dt_ux = -(ux_dx_ux + uy_dy_ux) - R_spec*T/rho*dx_rho + Fx;
            Real dt_uy = -(ux_dx_uy + uy_dy_uy) - R_spec*T/rho*dy_rho + Fy;
            #endif

            if(0)
            if(x == 1 && y == 2) {
                PRINT_VARS(vc);
                PRINT_VARS(vn);
                PRINT_VARS(vs);
                PRINT_VARS(ve);
                PRINT_VARS(vw);
                PRINT_VARS(vne);
                PRINT_VARS(vnw);
                PRINT_VARS(vse);
                PRINT_VARS(vsw);

                PRINT_F(dx);
                PRINT_F(dy);
                PRINT_F(ux);
                PRINT_F(uy);
                PRINT_F(dx_ux);
                PRINT_F(dx_uy);
                PRINT_F(dy_ux);
                PRINT_F(dy_uy);
                PRINT_F(dx_rho);
                PRINT_F(dy_rho);
                PRINT_F(dxx_ux);
                PRINT_F(dxx_uy);
                PRINT_F(dyy_ux);
                PRINT_F(dyy_uy);
                PRINT_F(dxy_ux);
                PRINT_F(dxy_uy);
                PRINT_F(ux_dx_ux);
                PRINT_F(ux_dx_uy);
                PRINT_F(uy_dy_ux);
                PRINT_F(uy_dy_uy);
                PRINT_F(ux_dx_rho);
                PRINT_F(uy_dy_rho);

                PRINT_F(dt_rho);
                PRINT_F(dt_ux);
                PRINT_F(dt_uy);
            }

            if(cflags) {
                if(cflags & SIM_SET_RHO) dt_rho = 0;
                if(cflags & SIM_SET_UX) dt_ux = 0;
                if(cflags & SIM_SET_UY) dt_uy = 0;
            }

            // dt_rho = 0;
            next_rhos[i] = rho + dt*dt_rho;
            next_uxs[i] = ux + dt*dt_ux;
            next_uys[i] = uy + dt*dt_uy;
        }
    );

    return dt;
}

extern "C" bool sim_make_flow_vertices(Sim_Flow_Vertex* vertices, Real* uxs, Real* uys, Draw_Lines_Config config)
{
    csize nx = (csize) config.nx;
    csize ny = (csize) config.ny;
    csize pix_size = config.pix_size;
    float pix_sizef = config.pix_size;

    cuda_for_2D(0, 0, nx/pix_size, ny/pix_size, [=]SHARED(csize xi, csize yi){
        xi *= pix_size;
        yi *= pix_size;
        csize i = xi + yi*nx;

        Real realux = 0;
        Real realuy = 0;
        for(csize ox = 0; ox < pix_size; ox++)
            for(csize oy = 0; oy < pix_size; oy++)
            {
                csize ic = (xi + ox) + (yi + oy)*nx;
                realux += uxs[ic];
                realuy += uys[ic];
            }
        
        float ux = (float) realux / (pix_sizef*pix_sizef);
        float uy = (float) realuy / (pix_sizef*pix_sizef);

        //normalize direction
        float len = hypotf(ux, uy);
        if(len > 0) {
            ux /= len;
            uy /= len;
        }

        //calculate variability
        #if 0
        Real variability = 0;
        for(csize ox = 0; ox < pix_size; ox++)
            for(csize oy = 0; oy < pix_size; oy++)
            {
                csize ic = (xi + ox) + (yi + oy)*nx;
                variability += fabs(ux*uxs[ic] + ux*uys[ic]);
            }
        variability = 1 - variability/(pix_sizef*pix_sizef);
        #endif

        float x = (xi + pix_sizef/2)*config.dx*2 - 1;
        float y = (yi + pix_sizef/2)*config.dy*2 - 1;

        float scaled_len = len*config.scale;
        if(scaled_len < config.min_size)
            scaled_len = config.min_size;
        if(scaled_len > config.max_size)
            scaled_len = config.max_size;
        if(len == 0)
            len = 1;

        float px = uy;
        float py = -ux;

        float ex = ux*scaled_len + x;
        float ey = uy*scaled_len + y;

        float v1x = x + px*config.width_i0;
        float v1y = y + py*config.width_i0;

        float v2x = x - px*config.width_i0;
        float v2y = y - py*config.width_i0;

        float v3x = ex + px*config.width_i1;
        float v3y = ey + py*config.width_i1;

        float v4x = ex - px*config.width_i1;
        float v4y = ey - py*config.width_i1;

        vertices[i*6+0] = Sim_Flow_Vertex{v1x, v1y, config.rgba_i0};
        vertices[i*6+1] = Sim_Flow_Vertex{v2x, v2y, config.rgba_i0};
        vertices[i*6+2] = Sim_Flow_Vertex{v3x, v3y, config.rgba_i1};
        vertices[i*6+3] = Sim_Flow_Vertex{v2x, v2y, config.rgba_i0};
        vertices[i*6+4] = Sim_Flow_Vertex{v3x, v3y, config.rgba_i1};
        vertices[i*6+5] = Sim_Flow_Vertex{v4x, v4y, config.rgba_i1};
    });
    return true;
}


extern "C" bool sim_run_tests()
{
    #ifdef TEST_CUDA_FOR_IMPL
    test_tiled_for((uint64_t) clock_ns());
    test_tiled_for_2D((uint64_t) clock_ns());
    #endif
    #ifdef TEST_CUDA_REDUCTION_IMPL
    test_reduce((uint64_t) clock_ns());
    #endif

    return true;
}