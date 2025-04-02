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


struct Vars_And_Flags {
    Sim_Flags flags;
    Real rho;
    Real ux;
    Real uy;
};

struct All_Vars {
    Sim_Flags flags;
    Vars val;
    Vars back_dx;
    Vars back_dy;
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
    Real lambda = params.second_viscosity;
    Real mu = params.dynamic_viscosity;
    
    Real R_spec = 287.052874;
    Real T = params.temperature;

    Cache_Tag tag = cache_tag_make();

    //TODO: try the second version of tiled for as well - should be faster more kernels where most of the time
    // is spent computing.

    #define PRINT_F(x) printf(#x " = %e\n", (double) (x))
    #define PRINT_I(x) printf(#x " = %lli\n", (long long) (x))

    #define PRINT_VARS(v) printf(#v " = {rho:%e ux:%e uy:%e}\n", (v).rho, (v).ux, (v).uy)

    //Precaclulate first order derivations
    //TODO: assymetric tiles
    #if 1
    All_Vars* all = cache_alloc(All_Vars, nx*ny, &tag);
    cuda_tiled_for_2D<1, 1, Vars_And_Flags>(0, 0, nx, ny, 
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) {
                csize i = x + y*nx;
                Sim_Flags cflags = flags[i];
                
                //TODO: early check
                // if(cflags) {
                    return Vars_And_Flags{
                        cflags,
                        cflags & SIM_SET_RHO ? set_rho[i] : prev_rhos[i],
                        cflags & SIM_SET_UX ? set_ux[i] : prev_uxs[i],
                        cflags & SIM_SET_UY ? set_uy[i] : prev_uys[i],
                    };
                // }
            }
            else
                return Vars_And_Flags{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Vars_And_Flags* __restrict__ shared){
            csize i = x + y*nx;
            Vars_And_Flags vc = shared[tx + ty*tile_size_x];
            
            // Vars_And_Flags vn = shared[tx + (ty+1)*tile_size_x];
            Vars_And_Flags vs = shared[tx + (ty-1)*tile_size_x];
            // Vars_And_Flags ve = shared[(tx+1) + ty*tile_size_x];
            Vars_And_Flags vw = shared[(tx-1) + ty*tile_size_x];

            All_Vars out;
            out.flags = vc.flags;
            out.val.ux = vc.ux;
            out.val.uy = vc.uy;
            out.val.rho = vc.rho;

            out.back_dx.rho = (vc.rho - vw.rho)/dx;
            out.back_dx.ux = (vc.ux - vw.ux)/dx;
            out.back_dx.uy = (vc.uy - vw.uy)/dx;

            out.back_dy.rho = (vc.rho - vs.rho)/dy;
            out.back_dy.ux = (vc.ux - vs.ux)/dy;
            out.back_dy.uy = (vc.uy - vs.uy)/dy;

            if(vc.flags) {
                if(vc.flags & SIM_SET_DX_RHO) out.back_dx.rho = set_dx_rho[i];
                if(vc.flags & SIM_SET_DX_UX) out.back_dx.ux = set_dx_ux[i];
                if(vc.flags & SIM_SET_DX_UY) out.back_dx.uy = set_dx_uy[i];
                if(vc.flags & SIM_SET_DY_RHO) out.back_dy.rho = set_dy_rho[i];
                if(vc.flags & SIM_SET_DY_UX) out.back_dy.ux = set_dy_ux[i];
                if(vc.flags & SIM_SET_DY_UY) out.back_dy.uy = set_dy_uy[i];
            }
            all[i] = out;
        }
    );

    cuda_tiled_for_2D<1, 1, All_Vars>(0, 0, nx, ny, 
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) 
                return all[x + y*nx];
            else
                return All_Vars{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, All_Vars* __restrict__ shared){
            csize i = x + y*nx;

            All_Vars vc = shared[tx + ty*tile_size_x];
            
            All_Vars vn = shared[tx + (ty+1)*tile_size_x];
            All_Vars vs = shared[tx + (ty-1)*tile_size_x];
            All_Vars ve = shared[(tx+1) + ty*tile_size_x];
            All_Vars vw = shared[(tx-1) + ty*tile_size_x];

            All_Vars vne = shared[(tx+1) + (ty+1)*tile_size_x];
            All_Vars vnw = shared[(tx-1) + (ty+1)*tile_size_x];
            All_Vars vse = shared[(tx+1) + (ty-1)*tile_size_x];
            All_Vars vsw = shared[(tx-1) + (ty-1)*tile_size_x];

            Real Fx = 0; 
            Real Fy = 0;

            Real rho = vc.val.rho;
            Real ux = vc.val.ux;
            Real uy = vc.val.uy;

            Real uw_dx_rho = ux < 0 || (vc.flags & SIM_SET_DX_RHO) ? vc.back_dx.rho : ve.back_dx.rho;
            Real uw_dx_ux = ux < 0 || (vc.flags & SIM_SET_DX_UX) ? vc.back_dx.ux : ve.back_dx.ux;
            Real uw_dx_uy = ux < 0 || (vc.flags & SIM_SET_DX_UY) ? vc.back_dx.uy : ve.back_dx.uy;

            Real uw_dy_rho = uy < 0 || (vc.flags & SIM_SET_DY_RHO) ? vc.back_dy.rho : vn.back_dy.rho;
            Real uw_dy_ux = uy < 0 || (vc.flags & SIM_SET_DY_UX) ? vc.back_dy.ux : vn.back_dy.ux;
            Real uw_dy_uy = uy < 0 || (vc.flags & SIM_SET_DY_UY) ? vc.back_dy.uy : vn.back_dy.uy;

            Real ct_dx_rho = (vc.flags & SIM_SET_DX_RHO) ? vc.back_dx.rho : (vc.back_dx.rho + ve.back_dx.rho)/2;
            Real ct_dx_ux = (vc.flags & SIM_SET_DX_UX) ? vc.back_dx.ux : (vc.back_dx.ux + ve.back_dx.ux)/2;
            Real ct_dx_uy = (vc.flags & SIM_SET_DX_UY) ? vc.back_dx.uy : (vc.back_dx.uy + ve.back_dx.uy)/2;

            Real ct_dy_rho = (vc.flags & SIM_SET_DY_RHO) ? vc.back_dy.rho : (vc.back_dy.rho + vn.back_dy.rho)/2;
            Real ct_dy_ux = (vc.flags & SIM_SET_DY_UX) ? vc.back_dy.ux : (vc.back_dy.ux + vn.back_dy.ux)/2;
            Real ct_dy_uy = (vc.flags & SIM_SET_DY_UY) ? vc.back_dy.uy : (vc.back_dy.uy + vn.back_dy.uy)/2;

            //(ve - vc) - (vc - vw) = ve - 2vc - vw
            Real ct_dxx_ux = (ve.back_dx.ux - vc.back_dx.ux)/dx;
            Real ct_dxx_uy = (ve.back_dx.uy - vc.back_dx.uy)/dx;

            Real ct_dyy_ux = (vn.back_dy.ux - vc.back_dy.ux)/dy;
            Real ct_dyy_uy = (vn.back_dy.uy - vc.back_dy.uy)/dy;

            Real ct_dxy_ux = (vn.back_dx.ux - vne.back_dx.ux - vs.back_dx.ux + vse.back_dx.ux)/(2*dy);
            Real ct_dxy_uy = (vn.back_dx.uy - vne.back_dx.uy - vs.back_dx.uy + vse.back_dx.uy)/(2*dy);

            if(0)
            if(x == 3 && y == 5) {
                PRINT_F(ct_dxx_ux);
                PRINT_F(ct_dxx_uy);

                PRINT_VARS(vc.val);
                PRINT_VARS(vn.val);
                PRINT_VARS(vs.val);
                PRINT_VARS(ve.val);
                PRINT_VARS(vw.val);
                PRINT_VARS(vne.val);
                PRINT_VARS(vnw.val);
                PRINT_VARS(vse.val);
                PRINT_VARS(vsw.val);

                PRINT_VARS(vc.back_dx);
                PRINT_VARS(vn.back_dx);
                PRINT_VARS(vs.back_dx);
                PRINT_VARS(ve.back_dx);
                PRINT_VARS(vw.back_dx);
                PRINT_VARS(vne.back_dx);
                PRINT_VARS(vnw.back_dx);
                PRINT_VARS(vse.back_dx);
                PRINT_VARS(vsw.back_dx);

                PRINT_VARS(vc.back_dy);
                PRINT_VARS(vn.back_dy);
                PRINT_VARS(vs.back_dy);
                PRINT_VARS(ve.back_dy);
                PRINT_VARS(vw.back_dy);
                PRINT_VARS(vne.back_dy);
                PRINT_VARS(vnw.back_dy);
                PRINT_VARS(vse.back_dy);
                PRINT_VARS(vsw.back_dy);
            }

            // Real dt_rho = -(ux*uw_dx_rho + uy*uw_dy_rho) - rho*(uw_dx_ux + uw_dy_uy);
            // Real dt_rho = -(ux*uw_dx_rho + uy*uw_dy_rho) - rho*(uw_dx_ux);
            // Real dt_rho = -(ux*uw_dx_rho + uy*uw_dy_rho) - rho*(ct_dx_ux + ct_dy_uy);
            Real dt_rho = -(ux*uw_dx_rho + uy*uw_dy_rho) - rho*(ct_dx_ux + ct_dy_uy);
            // Real dt_rho = -(ux*uw_dx_rho + uy*uw_dy_rho);
            Real dt_ux = -(ux*uw_dx_ux + uy*uw_dy_ux) - R_spec*T/rho*uw_dx_rho + Fx
                + 1/rho*((lambda + 2*mu)*ct_dxx_ux + (lambda + mu)*ct_dxy_uy + mu*ct_dyy_ux);
            Real dt_uy = -(ux*uw_dx_uy + uy*uw_dy_uy) - R_spec*T/rho*uw_dy_rho + Fy
                + 1/rho*((lambda + 2*mu)*ct_dyy_uy + (lambda + mu)*ct_dxy_ux + mu*ct_dxx_uy);

            // Real dt_rho = 0;
            // Real dt_ux = 0
            //     + 1/rho*((lambda + 2*mu)*ct_dxx_ux + (lambda + mu)*ct_dxy_uy + mu*ct_dyy_ux);
            // Real dt_uy = 0
            //     + 1/rho*((lambda + 2*mu)*ct_dyy_uy + (lambda + mu)*ct_dxy_ux + mu*ct_dxx_uy);

            if(vc.flags) {
                if(vc.flags & SIM_SET_RHO) dt_rho = 0;
                if(vc.flags & SIM_SET_UX) dt_ux = 0;
                if(vc.flags & SIM_SET_UY) dt_uy = 0;
            }

            next_rhos[i] = rho + dt*dt_rho;
            next_uxs[i] = ux + dt*dt_ux;
            next_uys[i] = uy + dt*dt_uy;
        }
    );

    #else
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
            
            Vars vn = shared[tx + (ty+1)*tile_size_x];
            Vars vs = shared[tx + (ty-1)*tile_size_x];
            Vars ve = shared[(tx+1) + ty*tile_size_x];
            Vars vw = shared[(tx-1) + ty*tile_size_x];

            Vars vnn = shared[tx + (ty+2)*tile_size_x];
            Vars vss = shared[tx + (ty-2)*tile_size_x];
            Vars vee = shared[(tx+2) + ty*tile_size_x];
            Vars vww = shared[(tx-2) + ty*tile_size_x];

            Vars vne = shared[(tx+1) + (ty+1)*tile_size_x];
            Vars vnw = shared[(tx-1) + (ty+1)*tile_size_x];
            Vars vse = shared[(tx+1) + (ty-1)*tile_size_x];
            Vars vsw = shared[(tx-1) + (ty-1)*tile_size_x];

            Real Fx = 0; 
            Real Fy = 0;

            Real rho = vc.rho;
            Real ux = vc.ux;
            Real uy = vc.uy;

            Real dx_rho = ux >= 0 ? (vc.rho - vw.rho)/dx : (ve.rho - vc.rho)/dx;
            Real dx_ux = ux >= 0 ? (vc.ux - vw.ux)/dx : (ve.ux - vc.ux)/dx;
            Real dx_uy = ux >= 0 ? (vc.uy - vw.uy)/dx : (ve.uy - vc.uy)/dx;

            Real dy_rho = uy >= 0 ? (vc.rho - vs.rho)/dy : (vn.rho - vc.rho)/dy;
            Real dy_ux = uy >= 0 ? (vc.ux - vs.ux)/dy : (vn.ux - vc.ux)/dy;
            Real dy_uy = uy >= 0 ? (vc.uy - vs.uy)/dy : (vn.uy - vc.uy)/dy;

            //second der should take into account boundary set dx,dy
            Real dxx_ux = (ve.ux + 2*vc.ux - vw.ux)/(dx*dx);
            Real dxx_uy = (ve.uy + 2*vc.uy - vw.uy)/(dx*dx);

            Real dyy_ux = (vn.ux + 2*vc.ux - vs.ux)/(dy*dy);
            Real dyy_uy = (vn.uy + 2*vc.uy - vs.uy)/(dy*dy);

            Real dxy_ux = (vne.ux - vnw.ux - vse.ux + vsw.ux)/(4*dx*dy);
            Real dxy_uy = (vne.uy - vnw.uy - vse.uy + vsw.uy)/(4*dx*dy);

            if(cflags) {
                if(cflags & SIM_SET_DX_UX) dx_ux = set_dx_ux[i];
                if(cflags & SIM_SET_DX_UY) dx_uy = set_dx_uy[i];
                if(cflags & SIM_SET_DY_UX) dy_ux = set_dy_ux[i];
                if(cflags & SIM_SET_DY_UY) dy_uy = set_dy_uy[i];
                if(cflags & SIM_SET_DX_RHO) dx_rho = set_dx_rho[i];
                if(cflags & SIM_SET_DY_RHO) dy_rho = set_dy_rho[i];
            }

            Real dt_rho = -(ux*dx_rho + uy*dy_rho) + rho*(dx_ux + dy_uy);
            Real dt_ux = -(ux*dx_ux + uy*dy_ux) - R_spec*T/rho*dx_rho + Fx
                + 1/rho*((lambda + 2*mu)*dxx_ux + (lambda + mu)*dxy_uy + mu*dyy_ux);
            Real dt_uy = -(ux*dx_uy + uy*dy_uy) - R_spec*T/rho*dy_rho + Fy
                + 1/rho*((lambda + 2*mu)*dyy_uy + (lambda + mu)*dxy_ux + mu*dxx_uy);

            if(0)
            if(x == 2 && y == 3) {
                PRINT_VARS(vc);
                PRINT_VARS(vn);
                PRINT_VARS(vs);
                PRINT_VARS(ve);
                PRINT_VARS(vw);
                PRINT_VARS(vne);
                PRINT_VARS(vnw);
                PRINT_VARS(vse);
                PRINT_VARS(vsw);

                PRINT_I(cflags);
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
    #endif

    cache_free(&tag);
    return dt;
}

SHARED uint32_t argb_to_hex(float r, float g, float b, float a)
{
    return (uint32_t) (r*255) << 16 | (uint32_t) (g*255) << 8 | (uint32_t) (b*255) << 0 | (uint32_t) ((1-a)*255) << 24;
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

        #define PI 3.14159265359

        //calculate gradient color
        float color_min_val = 1;
        float color_max_val = 100;
        uint32_t color = 0;
        if(isnan(len))
            color = 0xFF00FF; //Bright purple
        else if(1) {
            float val = min(max(val, color_min_val), color_max_val);
            float display = (val - color_min_val)/(color_max_val - color_min_val);
            color = argb_to_hex(display, display, display, 1);
        }
        else
        {
            float val = len;
            if(isnan(val))
                color = 0xFF00FF; //Bright purple
            else if(val < color_min_val)
            {
                float display = (1 - atan(color_min_val - val)/ PI*2)*0.3;
                color = argb_to_hex(display, display, display, 1);
                //Shades from dark gray to black

            }
            else if(val > color_max_val)
            {
                float display = (atan(val - color_min_val)/PI*2*0.3 + 0.7);
                color = argb_to_hex(display, display, display, 1);
                //Shades from bright gray to white
            }
            else
            {
                //Spectreum blue -> cyan -> green -> yellow -> red

                val = min(max(val, color_min_val), color_max_val- 0.0001);
                float d = color_max_val - color_min_val;
                val = d == 0.0 ? 0.5f : (val - color_min_val) / d;
                float m = 0.25f;
                float num = floor(val / m);
                float s = (val - num * m) / m;
                float r = 0, g = 0, b = 0;

                switch (int(num)) {
                    case 0 : r = 0; g = s; b = 1; break;
                    case 1 : r = 0; g = 1; b = 1-s; break;
                    case 2 : r = s; g = 1; b = 0; break;
                    case 3 : r = 1; g = 1-s; b = 0; break;
                }
                
                color = argb_to_hex(r, g, b, 1);
            }
        }

        vertices[i*6+0] = Sim_Flow_Vertex{v1x, v1y, color};
        vertices[i*6+1] = Sim_Flow_Vertex{v2x, v2y, color};
        vertices[i*6+2] = Sim_Flow_Vertex{v3x, v3y, color};
        vertices[i*6+3] = Sim_Flow_Vertex{v2x, v2y, color};
        vertices[i*6+4] = Sim_Flow_Vertex{v3x, v3y, color};
        vertices[i*6+5] = Sim_Flow_Vertex{v4x, v4y, color};
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