// #define USE_CUDA 0

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
    CUDA_TEST(cudaMalloc(&state->ro, bytes));
    CUDA_TEST(cudaMalloc(&state->ux, bytes));
    CUDA_TEST(cudaMalloc(&state->uy, bytes));
    return true;
}
extern "C" bool sim_const_state_init(Sim_Const_State* state, int32_t nx, int32_t ny)
{
    // sim_const_state_deinit(state);
    state->nx = nx;
    state->ny = ny;
    size_t cells = (size_t) nx * (size_t) ny;
    size_t faces_x = (size_t) (nx+1) * (size_t) ny; 
    size_t faces_y = (size_t) nx * (size_t) (ny+1); 

    CUDA_TEST(cudaMalloc(&state->cell_flags, cells * sizeof(Sim_Flags)));
    CUDA_TEST(cudaMalloc(&state->face_x_flags, cells * sizeof(Sim_Flags)));
    CUDA_TEST(cudaMalloc(&state->face_y_flags, cells * sizeof(Sim_Flags)));

    CUDA_TEST(cudaMalloc(&state->face_x_values, faces_x * sizeof(Sim_Face_Values)));
    CUDA_TEST(cudaMalloc(&state->face_y_values, faces_y * sizeof(Sim_Face_Values)));

    CUDA_TEST(cudaMalloc(&state->face_x_set_val_ro, faces_x * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_x_set_val_ux, faces_x * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_x_set_val_uy, faces_x * sizeof(Sim_Real)));

    CUDA_TEST(cudaMalloc(&state->face_x_set_der_ro, faces_x * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_x_set_der_ux, faces_x * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_x_set_der_uy, faces_x * sizeof(Sim_Real)));

    CUDA_TEST(cudaMalloc(&state->face_y_set_val_ro, faces_y * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_y_set_val_ux, faces_y * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_y_set_val_uy, faces_y * sizeof(Sim_Real)));
    
    CUDA_TEST(cudaMalloc(&state->face_y_set_der_ro, faces_y * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_y_set_der_ux, faces_y * sizeof(Sim_Real)));
    CUDA_TEST(cudaMalloc(&state->face_y_set_der_uy, faces_y * sizeof(Sim_Real)));

    return true;
}

extern "C" void sim_mut_state_deinit(Sim_Mut_State* state)
{
    CUDA_TEST(cudaFree(state->ro));
    CUDA_TEST(cudaFree(state->ux));
    CUDA_TEST(cudaFree(state->uy));
    memset(state, 0, sizeof *state);
}
extern "C" void sim_const_state_deinit(Sim_Const_State* state)
{
    ASSERT(false, "TODO");
    memset(state, 0, sizeof *state);
}

#undef CHECK_BOUNDS
// #define CHECK_BOUNDS(x, range) assert(0 <= (x) && (x) < (range))
#define CHECK_BOUNDS(x, range) \
    (0 <= (x) && (x) < (range)) ? 1 \
        : (printf("Bounds check failed %lli not in [0, %lli).\n", (long long) (x), (long long) (range)), \
            assert(!"bounds check failed"), 0)

template <typename T>
struct Span_2D {
    T* data;
    csize nx;
    csize ny;

    constexpr inline SHARED bool inside(csize x, csize y) const {
        return (0 <= x && x < nx && 0 <= y && y <= ny);
    }

    constexpr inline SHARED csize i(csize x, csize y) const {
        CHECK_BOUNDS(x, nx);
        CHECK_BOUNDS(y, ny);
        return x + y*nx;
    }

    constexpr inline SHARED T& at(csize x, csize y) const {
        CHECK_BOUNDS(x, nx);
        CHECK_BOUNDS(y, ny);
        return data[x + y*nx];
    }

    constexpr inline SHARED T& at(csize i) const {
        CHECK_BOUNDS(i, nx*ny);
        return data[i];
    }

    constexpr inline SHARED T& operator [](csize i) const {
        CHECK_BOUNDS(i, nx*ny);
        return data[i];
    }
};

template <typename T>
struct Span {
    T* data;
    csize size;

    constexpr inline SHARED T& at(csize i) const {
        CHECK_BOUNDS(i, size);
        return data[i];
    }

    constexpr inline SHARED T& operator [](csize i) const {
        CHECK_BOUNDS(i, size);
        return data[i];
    }
};

template <typename T>
Span_2D<T> face_x_span(T* data, csize nx, csize ny)
{
    return Span_2D<T>{data, nx+1, ny};
}

template <typename T>
Span_2D<T> face_y_span(T* data, csize nx, csize ny)
{
    return Span_2D<T>{data, nx, ny+1};
}

template <typename T>
Span_2D<T> span_cache_alloc(csize nx, Cache_Tag* tag)
{
    T* alloced = cache_alloc(T, nx, tag);
    return Span_2D<T>{alloced, nx};
}

template <typename T>
Span_2D<T> span_2d_cache_alloc(csize nx, csize ny, Cache_Tag* tag)
{
    T* alloced = cache_alloc(T, nx*ny, tag);
    return Span_2D<T>{alloced, nx, ny};
}

extern "C" double sim_step(Sim_Mut_State* next, const Sim_Mut_State* prev, Sim_Const_State* const_state, Sim_Params params)
{
    csize nx = next->nx;
    csize ny = next->ny;

    Span_2D<Real> prev_ros = {prev->ro, nx, ny};
    Span_2D<Real> prev_uxs = {prev->ux, nx, ny};
    Span_2D<Real> prev_uys = {prev->uy, nx, ny};

    Span_2D<Real> next_ros = {next->ro, nx, ny};
    Span_2D<Real> next_uxs = {next->ux, nx, ny};
    Span_2D<Real> next_uys = {next->uy, nx, ny};
    Span_2D<Sim_Flags> cell_flags = {const_state->cell_flags, nx, ny}; 

    Span_2D<Sim_Flags> face_x_flags = face_x_span(const_state->face_x_flags, nx, ny);
    Span_2D<Real> face_x_set_val_ro = face_x_span(const_state->face_x_set_val_ro, nx, ny);
    Span_2D<Real> face_x_set_val_ux = face_x_span(const_state->face_x_set_val_ux, nx, ny);
    Span_2D<Real> face_x_set_val_uy = face_x_span(const_state->face_x_set_val_uy, nx, ny);

    Span_2D<Real> face_x_set_der_ro = face_x_span(const_state->face_x_set_der_ro, nx, ny);
    Span_2D<Real> face_x_set_der_ux = face_x_span(const_state->face_x_set_der_ux, nx, ny);
    Span_2D<Real> face_x_set_der_uy = face_x_span(const_state->face_x_set_der_uy, nx, ny);

    Span_2D<Sim_Flags> face_y_flags = face_y_span(const_state->face_y_flags, nx, ny);
    Span_2D<Real> face_y_set_val_ro = face_y_span(const_state->face_y_set_val_ro, nx, ny);
    Span_2D<Real> face_y_set_val_ux = face_y_span(const_state->face_y_set_val_ux, nx, ny);
    Span_2D<Real> face_y_set_val_uy = face_y_span(const_state->face_y_set_val_uy, nx, ny);
    
    Span_2D<Real> face_y_set_der_ro = face_y_span(const_state->face_y_set_der_ro, nx, ny);
    Span_2D<Real> face_y_set_der_ux = face_y_span(const_state->face_y_set_der_ux, nx, ny);
    Span_2D<Real> face_y_set_der_uy = face_y_span(const_state->face_y_set_der_uy, nx, ny);

    Real dt = params.dt;
    Real dx = params.region_width / nx;
    Real dy = params.region_height / ny;
    Real lambda = params.second_viscosity;
    Real mu = params.dynamic_viscosity;
    
    Real R_spec = 287.052874 ;
    Real T = params.temperature;
    T = 293;
    
    #define PRINT_F(x) printf(#x " = %e\n", (double) (x))
    #define PRINT_I(x) printf(#x " = %lli\n", (long long) (x))
    #define PRINT_VARS(v) printf(#v " = {ro:%e ux:%e uy:%e}\n", (v).ro, (v).ux, (v).uy)

    Span_2D<Sim_Face_Values> faces_x = face_x_span(const_state->face_x_values, nx, ny);
    Span_2D<Sim_Face_Values> faces_y = face_y_span(const_state->face_y_values, nx, ny);
    
    //===============================================
    // calcultae face values and normal derivatives 
    //===============================================
    auto calc_face = []SHARED(Sim_Vars_And_Flags cn, Sim_Vars_And_Flags cp, Real dd, 
        Sim_Flags flags, bool is_x_face, csize x, csize y,
        Span_2D<Real> set_val_ro, Span_2D<Real> set_val_ux, Span_2D<Real> set_val_uy, 
        Span_2D<Real> set_der_ro, Span_2D<Real> set_der_ux, Span_2D<Real> set_der_uy) -> Sim_Face_Values
    {
        Sim_Face_Values out = {0};
        out.flags = flags;

        bool outside_n = !!(cn.flags & SIM_OUTSIDE_CELL);
        bool outside_p = !!(cp.flags & SIM_OUTSIDE_CELL);
        // if(outside_n && outside_p)
            // out.flags |= SIM_OUTSIDE_WALL;

        out.average.ro = (cp.ro + cn.ro)/2;
        out.average.ux = (cp.ux + cn.ux)/2;
        out.average.uy = (cp.uy + cn.uy)/2;

        Real upwind_by = is_x_face 
            ? out.average.ux
            : out.average.uy;

        out.upwind.ro = cn.ro;
        out.upwind.ux = cn.ux;
        out.upwind.uy = cn.uy;

        // out.upwind.ro = upwind_by < 0 ? cp.ro : cn.ro;
        // out.upwind.ux = upwind_by < 0 ? cp.ux : cn.ux;
        // out.upwind.uy = upwind_by < 0 ? cp.uy : cn.uy;

        Sim_Vars out_der = {0};
        out_der.ro = (cp.ro - cn.ro)/dd;
        out_der.ux = (cp.ux - cn.ux)/dd;
        out_der.uy = (cp.uy - cn.uy)/dd;


        if(y == 10 && x == 2) {
            int k = 0;
        }

        if(flags) {
            if(flags & SIM_SET_VAL_RO) {
                Real v = set_val_ro.at(x, y);
                out.average.ro = v;
                out.upwind.ro = v;

                out_der.ro = outside_n 
                    ? 2*(cp.ro - v)/dd
                    : 2*(v - cn.ro)/dd;
            }

            if(flags & SIM_SET_VAL_UX) {
                Real v = set_val_ux.at(x, y);
                out.average.ux = v;
                out.upwind.ux = v;

                out_der.ux = outside_n 
                    ? 2*(cp.ux - v)/dd
                    : 2*(v - cn.ux)/dd;
            }

            if(flags & SIM_SET_VAL_UY) {
                Real v = set_val_uy.at(x, y);
                out.average.uy = v;
                out.upwind.uy = v;

                out_der.uy = outside_n 
                    ? 2*(cp.uy - v)/dd
                    : 2*(v - cn.uy)/dd;
            }
            
            
            if(flags & SIM_SET_DER_RO) {
                Real der = set_der_ro.at(x, y);
                der = 0;
                out_der.ro = der;
                out.average.ro = outside_n 
                    ? cp.ro - der*dd/2
                    : der*dd/2 + cn.ro;
                out.upwind.ro = out.average.ro;
            }

            if(flags & SIM_SET_DER_UX) {
                Real der = set_der_ux.at(x, y);
                der = 0;
                out_der.ux = der;
                out.average.ux = outside_n 
                    ? cp.ux - der*dd/2
                    : der*dd/2 + cn.ux;
                out.upwind.ux = out.average.ux;
            }

            if(flags & SIM_SET_DER_UY) {
                Real der = set_der_uy.at(x, y);
                der = 0;
                out_der.uy = der;
                out.average.uy = outside_n 
                    ? cp.uy - der*dd/2
                    : der*dd/2 + cn.uy;
                out.upwind.uy = out.average.uy;
            }
        }

        if(is_x_face)
            out.der_x = out_der;
        else
            out.der_y = out_der;
        return out;
    };

    cuda_tiled_for_2D<1, 0, Sim_Vars_And_Flags>(0, 0, face_x_flags.nx, face_x_flags.ny, 
        [=]SHARED(csize x, csize y, csize face_nx, csize face_ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) {
                return Sim_Vars_And_Flags{
                    cell_flags.at(x, y),
                    prev_ros.at(x, y),
                    prev_uxs.at(x, y),
                    prev_uys.at(x, y),
                };
            }
            else
                return Sim_Vars_And_Flags{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Sim_Vars_And_Flags* __restrict__ shared){
            Sim_Vars_And_Flags ve = shared[tx + ty*tile_size_x];
            Sim_Vars_And_Flags vw = shared[tx-1 + ty*tile_size_x];

            Sim_Flags flags = face_x_flags.at(x, y);
            faces_x.at(x, y) = calc_face(vw, ve, dx, flags, true, x, y,
                face_x_set_val_ro, face_x_set_val_ux, face_x_set_val_uy, 
                face_x_set_der_ro, face_x_set_der_ux, face_x_set_der_uy
            ); 
        }
    );

    cuda_tiled_for_2D<0, 1, Sim_Vars_And_Flags>(0, 0, face_y_flags.nx, face_y_flags.ny, 
        [=]SHARED(csize x, csize y, csize face_nx, csize face_ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) {
                return Sim_Vars_And_Flags{
                    cell_flags.at(x, y),
                    prev_ros.at(x, y),
                    prev_uxs.at(x, y),
                    prev_uys.at(x, y),
                };
            }
            else
                return Sim_Vars_And_Flags{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Sim_Vars_And_Flags* __restrict__ shared){
            Sim_Vars_And_Flags vn = shared[tx + ty*tile_size_x];
            Sim_Vars_And_Flags vs = shared[tx + (ty-1)*tile_size_x];

            Sim_Flags flags = face_y_flags.at(x, y);
            faces_y.at(x, y) = calc_face(vs, vn, dy, flags, false, x, y,
                face_y_set_val_ro, face_y_set_val_ux, face_y_set_val_uy, 
                face_y_set_der_ro, face_y_set_der_ux, face_y_set_der_uy
            ); 
        }
    );
    
    //===============================================
    // caclulate face tangential derivatives 
    //===============================================
    if(0) {
    auto calc_face_tangential_der = []SHARED(
        Sim_Vars_And_Flags vn, Sim_Vars_And_Flags vc, Sim_Vars_And_Flags vp, Real dd, csize x, csize y,
        Span_2D<Real> set_der_ro, Span_2D<Real> set_der_ux, Span_2D<Real> set_der_uy
    ) -> Sim_Vars {
        Sim_Vars ders = {0};
        ders.ro = (vp.ro - vn.ro)/(2*dd);
        ders.ux = (vp.ux - vn.ux)/(2*dd);
        ders.uy = (vp.uy - vn.uy)/(2*dd);
        if((vp.flags | vn.flags | vc.flags) & SIM_OUTSIDE_WALL) {
            //if both are outside face or center is outside face set derivation to 
            if((vp.flags & vn.flags | vc.flags) & SIM_OUTSIDE_WALL) {
                ders = Sim_Vars{0};
            }
            else if(vp.flags & SIM_OUTSIDE_WALL ) {
                ders.ro = (vc.ro - vn.ro)/dd;
                ders.ux = (vc.ux - vn.ux)/dd;
                ders.uy = (vc.uy - vn.uy)/dd;
            }
            else if(vn.flags & SIM_OUTSIDE_WALL ) {
                ders.ro = (vp.ro - vc.ro)/dd;
                ders.ux = (vp.ux - vc.ux)/dd;
                ders.uy = (vp.uy - vc.uy)/dd;
            }
        }

        if(vc.flags) {
            if(vc.flags & SIM_SET_DER_RO) ders.ro = set_der_ro.at(x, y);
            if(vc.flags & SIM_SET_DER_UX) ders.ux = set_der_ux.at(x, y);
            if(vc.flags & SIM_SET_DER_UY) ders.uy = set_der_uy.at(x, y);
        }
        return ders;
    };

    cuda_tiled_for_2D<1, 0, Sim_Vars_And_Flags>(0, 0, faces_x.nx, faces_x.ny, 
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) {
                return Sim_Vars_And_Flags{
                    faces_x.at(x, y).flags,
                    faces_x.at(x, y).average.ro,
                    faces_x.at(x, y).average.ux,
                    faces_x.at(x, y).average.uy,
                };
            }
            else
                return Sim_Vars_And_Flags{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Sim_Vars_And_Flags* __restrict__ shared){
            Sim_Vars_And_Flags ve = shared[tx+1 + ty*tile_size_x];
            Sim_Vars_And_Flags vc = shared[tx+0 + ty*tile_size_x];
            Sim_Vars_And_Flags vw = shared[tx-1 + ty*tile_size_x];

            faces_x.at(x, y).der_y = calc_face_tangential_der(ve, vc, vw, dy, x, y, 
                face_x_set_der_ro, face_x_set_der_ux, face_x_set_der_uy 
            );
        }
    );

    cuda_tiled_for_2D<0, 1, Sim_Vars_And_Flags>(0, 0, faces_y.nx, faces_y.ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny) {
                return Sim_Vars_And_Flags{
                    faces_y.at(x, y).flags,
                    faces_y.at(x, y).average.ro,
                    faces_y.at(x, y).average.ux,
                    faces_y.at(x, y).average.uy,
                };
            }
            else
                return Sim_Vars_And_Flags{0};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Sim_Vars_And_Flags* __restrict__ shared){
            Sim_Vars_And_Flags ve = shared[tx + (ty+1)*tile_size_x];
            Sim_Vars_And_Flags vc = shared[tx + (ty+0)*tile_size_x];
            Sim_Vars_And_Flags vw = shared[tx + (ty-1)*tile_size_x];

            faces_y.at(x, y).der_y = calc_face_tangential_der(ve, vc, vw, dx, x, y, 
                face_y_set_der_ro, face_y_set_der_ux, face_y_set_der_uy 
            );
        }
    );
    }

    //===============================================
    // Perform step
    //===============================================
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(csize x, csize y){
        Real Fx = 0; 
        Real Fy = 0;

        Real ro = prev_ros.at(x, y);
        Real ux = prev_uxs.at(x, y);
        Real uy = prev_uys.at(x, y);
        Sim_Flags flags = cell_flags.at(x, y);

        Sim_Face_Values fw = faces_x.at(x+0, y);
        Sim_Face_Values fe = faces_x.at(x+1, y);
        Sim_Face_Values fs = faces_y.at(x, y+0);
        Sim_Face_Values fn = faces_y.at(x, y+1);

        if(y == ny/2 && x == 1) {
            int k = 0;
        }

        Real dt_ro = 
            -(fe.average.ro*fe.average.ux - fw.average.ro*fw.average.ux)/dx 
            // -(fn.average.ro*fn.average.uy - fs.average.ro*fs.average.uy)/dy
        ;

        // dt_ro = 0;
        Real ro_dt_ux = 
            -(fe.upwind.ro*fe.upwind.ux*fe.average.ux - fw.upwind.ro*fw.upwind.ux*fw.average.ux)/dx
            // -(fn.upwind.ro*fn.upwind.ux*fn.average.uy - fs.upwind.ro*fs.upwind.ux*fs.average.uy)/dy
            // +(lambda + 2*mu)*(fe.der_x.ux - fw.der_x.ux)/dx 
            // +lambda*(fe.der_y.uy - fw.der_y.uy)/dx 
            // +mu*(fn.der_x.ux - fs.der_x.ux)/dy 
            // +mu*(fn.der_y.uy - fs.der_y.uy)/dy
            // -R_spec*T*(fe.upwind.ro - fw.upwind.ro)/dx 
            // -ux*dt_ro + ro*Fx
        ;

        // Real ro_dt_ux = 
        //     -(fe.upwind.ro*fe.upwind.ux*fe.average.ux - fw.upwind.ro*fw.upwind.ux*fw.average.ux)/dx
        //     -(fn.upwind.ro*fn.upwind.ux*fn.average.uy - fs.upwind.ro*fs.upwind.ux*fs.average.uy)/dy
        //     +(lambda + 2*mu)*(fe.der_x.ux - fw.der_x.ux)/dx + lambda*(fe.der_y.uy - fw.der_y.uy)/dx 
        //     +mu*(fn.der_x.ux - fs.der_x.ux)/dy + mu*(fn.der_y.uy - fs.der_y.uy)/dy
        //     -R_spec*T*(fe.average.ro - fw.average.ro)/dx - ux*dt_ro + ro*Fx;
    
        Real ro_dt_uy = 
            -(fe.upwind.ro*fe.upwind.uy*fe.average.uy - fw.upwind.ro*fw.upwind.uy*fw.average.uy)/dy
            -(fn.upwind.ro*fn.upwind.uy*fn.average.ux - fs.upwind.ro*fs.upwind.uy*fs.average.ux)/dx
            +(lambda + 2*mu)*(fe.der_y.uy - fw.der_y.uy)/dy + lambda*(fe.der_x.ux - fw.der_x.ux)/dy 
            +mu*(fn.der_y.uy - fs.der_y.uy)/dx + mu*(fn.der_x.ux - fs.der_x.ux)/dx
            -R_spec*T*(fe.average.ro - fw.average.ro)/dy - uy*dt_ro + ro*Fy;

        Real dt_ux = ro_dt_ux/ro;
        Real dt_uy = ro_dt_uy/ro;
        
        // dt_ro = 0;
        // dt_ux = 0;
        dt_uy = 0;

        if(flags & SIM_OUTSIDE_CELL) {
            dt_ro = INFINITY;
            dt_ux = INFINITY;
            dt_uy = INFINITY;
        }

        next_ros.at(x, y) = ro + dt*dt_ro;
        next_uxs.at(x, y) = ux + dt*dt_ux;
        next_uys.at(x, y) = uy + dt*dt_uy;
    });

    return dt;
}

SHARED uint32_t argb_to_hex(float r, float g, float b, float a)
{
    return (uint32_t) (r*255) << 16 | (uint32_t) (g*255) << 8 | (uint32_t) (b*255) << 0 | (uint32_t) ((1-a)*255) << 24;
}

SHARED float4 hex_to_argb(uint32_t argb_hex) {

    uint32_t r = (argb_hex >> 16) & 0xFFu;
    uint32_t g = (argb_hex >> 8) & 0xFFu;
    uint32_t b = (argb_hex >> 0) & 0xFFu;
    uint32_t a = 255 - ((argb_hex >> 24) & 0xFFu);
    float4 argb = {r/255.0f, g/255.0f, b/255.0f, a/255.0f};
    return argb;
}

SHARED float4 val_to_rgba(float val, float min_val, float max_val)
{
    float pi = 3.14159265359f;
    if(isnan(val))
    {
        return float4{1, 0, 1, 1}; //Bright purple
    }
    else if(val < min_val)
    {
        //Shades from dark gray to black
        float display = (1 - atan(min_val - val)/pi*2)*0.3f;
        return float4{display, display, display, 1};

    }
    else if(val > max_val)
    {
        //Shades from bright gray to white
        float display = (atan(val - min_val)/pi*2*0.3f + 0.7f);
        return float4{display, display, display, 1};
    }
    else
    {
        //Spectreum blue -> cyan -> green -> yellow -> red
        val = min(max(val, min_val), max_val- 0.0001f);
        float d = max_val - min_val;
        val = d == 0 ? 0.5f : (val - min_val) / d;
        float m = 0.25f;
        float num = floor(val / m);
        float s = (val - num * m) / m;
        float r = 0, g = 0, b = 0;

        switch (int(num)) {
            case 0: r = 0; g = s; b = 1; break;
            case 1: r = 0; g = 1; b = 1-s; break;
            case 2: r = s; g = 1; b = 0; break;
            case 3: r = 1; g = 1-s; b = 0; break;
        }
        
        return float4{r, g, b, 1};
    }

}

extern "C" bool sim_make_flow_vertices(Sim_Color_Vertex* vertices, isize* out_count, isize capacity, Real* uxs, Real* uys, Draw_Lines_Config config)
{
    csize nx = (csize) config.nx;
    csize ny = (csize) config.ny;
    csize pix_size = config.pix_size;
    float pix_sizef = config.pix_size;
    csize cap = (csize) capacity;

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

        //calculate gradient color
        float color_min_val = 1;
        float color_max_val = 100;
        uint32_t color = 0;
        if(isnan(len))
            color = 0xFF00FF; //Bright purple
        else {
            float val = len;
            val = min(max(val, color_min_val), color_max_val);
            float display = (val - color_min_val)/(color_max_val - color_min_val);
            color = argb_to_hex(display, display, display, 1);
        }

        if(i+5 < cap) {
            vertices[i*6+0] = Sim_Color_Vertex{v1x, v1y, color};
            vertices[i*6+1] = Sim_Color_Vertex{v2x, v2y, color};
            vertices[i*6+2] = Sim_Color_Vertex{v3x, v3y, color};
            vertices[i*6+3] = Sim_Color_Vertex{v2x, v2y, color};
            vertices[i*6+4] = Sim_Color_Vertex{v3x, v3y, color};
            vertices[i*6+5] = Sim_Color_Vertex{v4x, v4y, color};
        }
    });

    *out_count = MIN(capacity, (isize)nx*ny*6);
    return true;
}

SHARED void* strided_2d_span_at(Strided_2D_Span span, csize x, csize y)
{
    CHECK_BOUNDS(x, span.nx);
    CHECK_BOUNDS(y, span.ny);

    uint8_t* ptr = (uint8_t*) span.data + x*span.stride + y*span.pitch;
    return ptr; 
}

extern "C" bool sim_make_face_vertices(Sim_Color_Vertex* vertices, isize* out_count, isize capacity, Strided_2D_Span face_values, Draw_Walls_Params params)
{
    float dx = params.dx;
    float dy = params.dy;

    float sx = params.is_x_dir ? params.line_width/2 : dx/2 + params.line_width/2;
    float sy = params.is_x_dir ? dy/2 + params.line_width/2 : params.line_width/2;

    float x_off = params.is_x_dir ? 0 : 0.5f;
    float y_off = params.is_x_dir ? 0.5f : 0; 
    csize cap = (csize) capacity;

    sx *= 2;
    sy *= 2;

    cuda_for_2D(0, 0, face_values.nx, face_values.ny, [=]SHARED(csize xi, csize yi){
        float val = (float) *(Sim_Real*) strided_2d_span_at(face_values, xi, yi);
        float4 color_f = val_to_rgba(val, params.min_val, params.max_val);
        uint color = argb_to_hex(color_f.x, color_f.y, color_f.z, 0);

        float x = dx*(xi + x_off)*2 - 1;
        float y = dy*(yi + y_off)*2 - 1;

        if(yi == 10 && xi == 2) {
            int k = 0;
        }

        float v1x = x - sx;
        float v1y = y - sy;

        float v2x = x + sx;
        float v2y = y - sy;

        float v3x = x + sx;
        float v3y = y + sy;

        float v4x = x - sx;
        float v4y = y + sy;

        csize i = xi + yi*face_values.nx;
        if(6*i+5 < cap) {
            vertices[i*6+0] = Sim_Color_Vertex{v1x, v1y, color};
            vertices[i*6+1] = Sim_Color_Vertex{v2x, v2y, color};
            vertices[i*6+2] = Sim_Color_Vertex{v3x, v3y, color};
            vertices[i*6+3] = Sim_Color_Vertex{v3x, v3y, color};
            vertices[i*6+4] = Sim_Color_Vertex{v1x, v1y, color};
            vertices[i*6+5] = Sim_Color_Vertex{v4x, v4y, color};
        }
        else {
            ASSERT(false);
        }
    });

    *out_count = MIN(capacity, (isize)face_values.nx*face_values.ny*6);
    return true;
}

SHARED int atomic_add_cpu_cuda(int *address, int val)
{
    #ifdef __CUDA_ARCH__
    return atomicAdd(address, val);
    #else
    int before = *address;
    *address += val;
    return before;
    #endif
}

static void test_hex_to_argb(uint32_t argb_hex, float r, float g, float b, float a)
{
    float4 rgba = hex_to_argb(argb_hex);
    TEST(rgba.x == r);
    TEST(rgba.y == g);
    TEST(rgba.z == b);
    TEST(rgba.w == a);

    uint32_t new_hex = argb_to_hex(r, g, b, a);
    TEST(new_hex == argb_hex);
}

extern "C" bool sim_make_face_vertices_flagged(
    Sim_Color_Vertex* vertices, isize* out_count, isize capacity, 
    Strided_2D_Span face_values, Strided_2D_Span face_flags, Sim_Flags flags_mask, Draw_Walls_Params params)
{
    csize off = (csize) *out_count;
    csize cap = capacity - off;
    static thread_local csize* t_count_ptr = NULL;
    if(t_count_ptr == NULL) 
        cudaMalloc(&t_count_ptr, sizeof *t_count_ptr);
    
    csize* count_ptr = t_count_ptr;
    cudaMemset(count_ptr, 0, sizeof *count_ptr);

    float dx = params.dx;
    float dy = params.dy;

    float sx = params.is_x_dir ? params.line_width/2 : dx/2 + params.line_width/2;
    float sy = params.is_x_dir ? dy/2 + params.line_width/2 : params.line_width/2;

    float x_off = params.is_x_dir ? 0 : 0.5f;
    float y_off = params.is_x_dir ? 0.5f : 0; 

    Sim_Face_Values* values = (Sim_Face_Values*) face_values.data;
    csize values_count = face_values.nx*face_values.ny;

    if(params.use_static_color == false) 
        params.use_static_color = face_values.data == NULL;

    sx *= 2;
    sy *= 2;

    cuda_for_2D(0, 0, face_values.nx, face_values.ny, [=]SHARED(csize xi, csize yi){
        Sim_Flags flags = *(Sim_Flags*) strided_2d_span_at(face_flags, xi, yi);

        if(flags & flags_mask) 
        {
            csize i = atomic_add_cpu_cuda(count_ptr, 6);
            if(i+5 < cap) {
                uint color = params.static_color;
                if(params.use_static_color == false) {
                    float val = (float) *(Sim_Real*) strided_2d_span_at(face_values, xi, yi);
                    float4 color_f = val_to_rgba(val, params.min_val, params.max_val);
                    color = argb_to_hex(color_f.x, color_f.y, color_f.z, 1);
                }

                float x = dx*(xi + x_off)*2 - 1;
                float y = dy*(yi + y_off)*2 - 1;

                float v1x = x - sx;
                float v1y = y - sy;

                float v2x = x + sx;
                float v2y = y - sy;

                float v3x = x + sx;
                float v3y = y + sy;

                float v4x = x - sx;
                float v4y = y + sy;

                vertices[off + i+0] = Sim_Color_Vertex{v1x, v1y, color};
                vertices[off + i+1] = Sim_Color_Vertex{v2x, v2y, color};
                vertices[off + i+2] = Sim_Color_Vertex{v3x, v3y, color};
                vertices[off + i+3] = Sim_Color_Vertex{v3x, v3y, color};
                vertices[off + i+4] = Sim_Color_Vertex{v1x, v1y, color};
                vertices[off + i+5] = Sim_Color_Vertex{v4x, v4y, color};
            }
            else {
                ASSERT(false);
            }
        } 
    });
    
    csize count = 0;
    cudaMemcpy(&count, count_ptr, sizeof *count_ptr, cudaMemcpyDeviceToHost);
    if(count + *out_count < capacity)
        *out_count += count;
    else
        *out_count = capacity;

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
