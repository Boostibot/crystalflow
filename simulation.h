#pragma once
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef USE_FLOATS
    typedef float Sim_Real;
#else
    typedef double Sim_Real;
#endif

typedef uint16_t Sim_Flags;

typedef enum Sim_Solver_Type{
    SOLVER_TYPE_NONE = 0,
    SOLVER_TYPE_NAIVE_EULER,
    SOLVER_TYPE_ENUM_COUNT,
} Sim_Solver_Type;

enum {
    SIM_SET_UX    = 1 << 0,
    SIM_SET_DX_UX = 1 << 1,
    SIM_SET_DY_UX = 1 << 2,

    SIM_SET_UY    = 1 << 3,
    SIM_SET_DX_UY = 1 << 4,
    SIM_SET_DY_UY = 1 << 5,

    SIM_SET_RHO    = 1 << 6,
    SIM_SET_DX_RHO = 1 << 7,
    SIM_SET_DY_RHO = 1 << 8,

    SIM_SET_VALS = 0
        | SIM_SET_UX
        | SIM_SET_UY
        | SIM_SET_RHO,

    SIM_SET_DERS = 0
        | SIM_SET_DX_UX | SIM_SET_DY_UX
        | SIM_SET_DX_UY | SIM_SET_DY_UY
        | SIM_SET_DX_RHO | SIM_SET_DY_RHO,
};

typedef struct Sim_Mut_State {
    int32_t nx;
    int32_t ny;

    Sim_Real* rho;
    Sim_Real* ux;
    Sim_Real* uy;
} Sim_Mut_State;

typedef struct Sim_Debug_Map {
    Sim_Real* data;
    char name[128];
} Sim_Debug_Map;

typedef struct Sim_Norms {
    double L1;
    double L2;
    double max;
    double min;
    char name[128];
} Sim_Norms;

typedef struct Sim_Const_State {
    int32_t nx;
    int32_t ny;

    //TODO: compress?
    Sim_Flags* flags;
    Sim_Real* set_rho;
    Sim_Real* set_ux;
    Sim_Real* set_uy;

    Sim_Real* set_dx_rho;
    Sim_Real* set_dx_ux;
    Sim_Real* set_dx_uy;

    Sim_Real* set_dy_rho;
    Sim_Real* set_dy_ux;
    Sim_Real* set_dy_uy;

    Sim_Debug_Map* debug_maps;
    int32_t debug_maps_count;
    int32_t debug_maps_capacity;

    Sim_Norms norms[32];
    int32_t norms_count;
    int32_t norms_capacity;
} Sim_Const_State; 

typedef struct Sim_Params {
    int64_t iter;
    double time;
    Sim_Solver_Type solver;

    double region_width;
    double region_height;
    double second_viscosity;
    double dynamic_viscosity;
    double temperature;
    double default_density;

    double dt;
    double max_dt;
    double min_dt;

    bool do_debug;
    bool do_stats;
    bool do_prints;
} Sim_Params;

extern "C" bool sim_mut_state_init(Sim_Mut_State* state, int32_t nx, int32_t ny);
extern "C" bool sim_const_state_init(Sim_Const_State* state, int32_t nx, int32_t ny);

extern "C" void sim_mut_state_deinit(Sim_Mut_State* state);
extern "C" void sim_const_state_deinit(Sim_Const_State* state);

extern "C" bool sim_step(Sim_Mut_State* next, const Sim_Mut_State* prev, Sim_Const_State* const_state, Sim_Params params);

typedef enum {
    MODIFY_UPLOAD,
    MODIFY_DOWNLOAD,
} Sim_Modify;

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify);
extern "C" void sim_modify_float(Sim_Real* device_memory, float* host_memory, size_t count, Sim_Modify modify);
extern "C" void sim_modify_double(Sim_Real* device_memory, double* host_memory, size_t count, Sim_Modify modify);

extern "C" bool sim_run_tests();
extern "C" bool sim_run_benchmarks(int N);

static const char* solver_type_to_cstring(Sim_Solver_Type type)
{
    switch(type)
    {
        default: return "unknown";
        case SOLVER_TYPE_NONE: return "none";
        case SOLVER_TYPE_NAIVE_EULER: return "naive-euler";
    }
}