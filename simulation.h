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

typedef int csize;
typedef uint16_t Sim_Flags;

typedef enum Sim_Solver_Type{
    SOLVER_TYPE_NONE = 0,
    SOLVER_TYPE_NAIVE_EULER,
    SOLVER_TYPE_ENUM_COUNT,
} Sim_Solver_Type;

enum {
    SIM_SET_VAL_RO = 1 << 0,
    SIM_SET_VAL_UX = 1 << 1,
    SIM_SET_VAL_UY = 1 << 2,

    SIM_SET_DER_RO = 1 << 3,
    SIM_SET_DER_UX = 1 << 4,
    SIM_SET_DER_UY = 1 << 5,

    SIM_SET_VALS = 0
        | SIM_SET_VAL_UX
        | SIM_SET_VAL_UY
        | SIM_SET_VAL_RO,

    SIM_SET_DERS = 0
        | SIM_SET_DER_UX
        | SIM_SET_DER_UY
        | SIM_SET_DER_RO,

    SIM_OUTSIDE_WALL = 1 << 6,
    SIM_DONT_SIMULATE = 1 << 15,
};

enum {
    SIM_OUTSIDE_CELL = 1,
};

typedef struct Sim_Mut_State {
    int32_t nx;
    int32_t ny;

    Sim_Real* ro;
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

typedef struct Sim_Vars {
    Sim_Real ro;
    Sim_Real ux;
    Sim_Real uy;
} Sim_Vars;

typedef struct Sim_Vars_And_Flags {
    Sim_Flags flags;
    Sim_Real ro;
    Sim_Real ux;
    Sim_Real uy;
} Sim_Vars_And_Flags;

typedef struct Sim_Face_Values {
    Sim_Flags flags;
    Sim_Vars average;
    Sim_Vars upwind;
    Sim_Vars der_x;
    Sim_Vars der_y;
} Sim_Face_Values;

typedef struct Sim_Const_State {
    int32_t nx;
    int32_t ny;

    //TODO: compress?
    Sim_Flags* cell_flags;
    Sim_Flags* face_x_flags;
    Sim_Flags* face_y_flags;

    Sim_Real* face_x_set_val_ro;
    Sim_Real* face_x_set_val_ux;
    Sim_Real* face_x_set_val_uy;

    Sim_Real* face_x_set_der_ro;
    Sim_Real* face_x_set_der_ux;
    Sim_Real* face_x_set_der_uy;

    Sim_Real* face_y_set_val_ro;
    Sim_Real* face_y_set_val_ux;
    Sim_Real* face_y_set_val_uy;
    
    Sim_Real* face_y_set_der_ro;
    Sim_Real* face_y_set_der_ux;
    Sim_Real* face_y_set_der_uy;

    Sim_Debug_Map* debug_maps;
    int32_t debug_maps_count;
    int32_t debug_maps_capacity;

    Sim_Face_Values* face_x_values;
    Sim_Face_Values* face_y_values;
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

extern "C" double sim_step(Sim_Mut_State* next, const Sim_Mut_State* prev, Sim_Const_State* const_state, Sim_Params params);

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

typedef struct Sim_Color_Vertex {
    float x;
    float y;
    uint32_t packed_color;
} Sim_Color_Vertex;

typedef struct Draw_Lines_Config {
    isize nx;
    isize ny;
    float dx;
    float dy;

    isize pix_size;

    float scale;
    float max_size;
    float min_size;

    float width_i0;
    float width_i1;

    uint32_t rgba_i0;
    uint32_t rgba_i1;

} Draw_Lines_Config;

typedef struct Strided_2D_Span {
    void* data;
    csize nx;
    csize ny;

    csize stride;
    csize pitch;
} Strided_2D_Span;

typedef struct Draw_Walls_Params {
    bool is_x_dir;
    bool use_static_color;

    float dx;
    float dy;
    float line_width;
    float min_val;
    float max_val;
    uint32_t static_color;
} Draw_Walls_Params;

extern "C" bool sim_make_flow_vertices(Sim_Color_Vertex* vertices, isize* out_count, isize capacity, Sim_Real* uxs, Sim_Real* uys, Draw_Lines_Config config);
extern "C" bool sim_make_face_vertices(Sim_Color_Vertex* vertices, isize* out_count, isize capacity, Strided_2D_Span face_values, Draw_Walls_Params params);
extern "C" bool sim_make_face_vertices_flagged(
    Sim_Color_Vertex* vertices, isize* out_count, isize capacity, 
    Strided_2D_Span face_values, Strided_2D_Span face_flags, Sim_Flags flags_mask, Draw_Walls_Params params);
