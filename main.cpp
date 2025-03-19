#define JOT_ALL_IMPL
#define COMPILE_GRAPHICS

#include "config.h"
#include "defines.h"
#include "simulation.h"
#include "log.h"
#include "assert.h"

#include "assert.h"
#include <cmath>
#include <stddef.h>
#include <cuda_runtime.h> 
#include <vector>

#define WINDOW_TITLE                "sim"
#define DEF_WINDOW_WIDTH	        1024 
#define DEF_WINDOW_HEIGHT	        1024

#define FPS_DISPLAY_PERIOD          0.0
#define SCREEN_UPDATE_PERIOD        0.1
#define SCREEN_UPDATE_IDLE_PERIOD   0.1
#define POLL_EVENTS_PERIOD          0.1
#define FREE_RUN_PERIOD             0.001

typedef Sim_Real Real;
static double clock_s();
static void wait_s(double seconds);

struct App_State {
    time_t init_time;
    bool is_in_step_mode;
    bool is_in_debug_mode;
    double remaining_steps;
    double step_by;
    double sim_time;
    double real_dt;
    double last_stats_save;
    int64_t iter;
    int64_t save_index;
    int32_t render_target;

    int32_t count_written_snapshots;
    int32_t count_written_stats;
    int32_t count_written_configs;

    Sim_Const_State gpu_const_state;
    Sim_Mut_State gpu_mut_states[2];

    Sim_Const_State cpu_const_state;
    Sim_Mut_State cpu_mut_state;

    Sim_Config config;
};

typedef int64_t isize;
void sim_set_wall_noslip_at(Sim_Const_State* cpu, int32_t x, int32_t y)
{
    int32_t nx = cpu->nx;
    int32_t ny = cpu->ny;
    CHECK_BOUNDS(x, nx);
    CHECK_BOUNDS(y, ny);
    for(int32_t dy = -1; dy <= 1; dy++)
        for(int32_t dx = -1; dx <= 1; dx++)
        {
            int32_t cx = x + dx;
            int32_t cy = y + dy;
            if(0 <= cx && cx < nx)
                if(0 <= cy && cy < ny)
                {
                    isize i = (isize) cx + (isize) cy*nx;
                    cpu->flags[i] = SIM_SET_UX | SIM_SET_UY;
                    cpu->set_ux[i] = 0;
                    cpu->set_uy[i] = 0;
                }
        }

    isize i = (isize) x + (isize) y*nx;
    cpu->flags[i] = SIM_SET_DERS | SIM_SET_VALS;
    cpu->set_rho[i] = 0;
    cpu->set_ux[i] = 0;
    cpu->set_uy[i] = 0;

    cpu->set_dx_rho[i] = 0;
    cpu->set_dx_ux[i] = 0;
    cpu->set_dx_uy[i] = 0;

    cpu->set_dy_rho[i] = 0;
    cpu->set_dy_ux[i] = 0;
    cpu->set_dy_uy[i] = 0;
}

void sim_set_boundary_conditions_tube(Sim_Const_State* cpu, Real intake, Real outake, Real intake_rho, Real outake_rho, const Sim_Config& config)
{
    //place spherical wall
    double obstacle_diam = config.region_height/10;
    double obstacle_x = config.region_width/3;
    double obstacle_y = config.region_height/2;
    double obstacle_diam2 = obstacle_diam*obstacle_diam;

    int32_t nx = cpu->nx;
    int32_t ny = cpu->ny;
    if(0)
    for(int32_t y = 0; y < ny; y++)
    {
        for(int32_t x = 0; x < nx; x++)
        {
            double posx = (x+0.5)/nx * config.region_width;
            double posy = (x+0.5)/ny * config.region_height;

            double diffx = posx - obstacle_x - config.region_width/2;
            double diffy = posy - obstacle_y - config.region_height/2;
            if(diffx*diffx - diffy*diffy <= obstacle_diam2)
                sim_set_wall_noslip_at(cpu, x, y);
        }
    }

    //input/output is dirichlet
    for(int32_t y = 0; y < ny; y++) {
        isize i0 = (isize) 0 + (isize) y*nx;
        isize i1 = (isize) (nx-1) + (isize) y*nx;

        cpu->flags[i0] = SIM_SET_VALS;
        cpu->set_ux[i0] = intake;
        cpu->set_uy[i0] = 0;
        cpu->set_rho[i0] = intake_rho;

        cpu->flags[i1] = SIM_SET_DERS;
        cpu->set_dx_ux[i1] = 0;
        cpu->set_dx_uy[i1] = 0;
        cpu->set_dx_rho[i1] = 0;
        // cpu->set_dy_ux[i1] = 0;
        // cpu->set_dy_uy[i1] = 0;
        // cpu->set_dy_rho[i1] = 0;
    }

    //set walls
    for(int32_t x = 0; x < nx; x++)
        sim_set_wall_noslip_at(cpu, x, 0);

    for(int32_t x = 0; x < nx; x++)
        sim_set_wall_noslip_at(cpu, x, ny - 1);
}

void sim_set_constant_initial_conditions(Sim_Mut_State* cpu, Real rho, Real ux, Real uy, const Sim_Config& config)
{
    int32_t nx = cpu->nx;
    int32_t ny = cpu->ny;
    for(int32_t y = 0; y < ny; y++)
        for(int32_t x = 0; x < nx; x++)
        {
            isize i = (isize) x + (isize) y*nx;
            cpu->ux[i] = ux;
            cpu->uy[i] = uy;
            cpu->rho[i] = rho;
        }
}

Sim_Mut_State cpu_sim_mut_state_init(int32_t nx, int32_t ny)
{
    size_t bytes = (size_t) nx* (size_t) ny*sizeof(Real);
    Sim_Mut_State state = {0};
    state.nx = nx;
    state.ny = ny;
    state.rho = (Real*) calloc(1, bytes); 
    state.ux = (Real*) calloc(1, bytes); 
    state.uy = (Real*) calloc(1, bytes); 
    return state;
}

void cpu_sim_mut_state_deinit(Sim_Mut_State* state)
{
    free(state->rho);
    free(state->ux);
    free(state->uy);
    memset(state, 0, sizeof *state);
}

Sim_Const_State cpu_sim_const_state_init(int32_t nx, int32_t ny)
{
    size_t bytes = (size_t) nx* (size_t) ny*sizeof(Real);
    Sim_Const_State state = {0};
    state.nx = nx;
    state.ny = ny;
    state.flags = (Sim_Flags*) calloc(1, (size_t) nx* (size_t) ny*sizeof(Sim_Flags)); 
    state.set_rho = (Real*) calloc(1, bytes); 
    state.set_ux = (Real*) calloc(1, bytes); 
    state.set_uy = (Real*) calloc(1, bytes); 

    state.set_dx_rho = (Real*) calloc(1, bytes); 
    state.set_dx_ux = (Real*) calloc(1, bytes); 
    state.set_dx_uy = (Real*) calloc(1, bytes); 

    state.set_dy_rho = (Real*) calloc(1, bytes); 
    state.set_dy_ux = (Real*) calloc(1, bytes); 
    state.set_dy_uy = (Real*) calloc(1, bytes); 
    return state;
}

void cpu_sim_const_state_deinit(Sim_Const_State* state)
{
    free(state->set_rho);
    free(state->set_ux);
    free(state->set_uy);

    free(state->set_dx_rho);
    free(state->set_dx_ux);
    free(state->set_dx_uy);

    free(state->set_dy_rho);
    free(state->set_dy_ux);
    free(state->set_dy_uy);
    memset(state, 0, sizeof *state);
}

Sim_Params sim_params_from_config(double time, int64_t iter, Sim_Config const& config) {
    Sim_Params params = {0};
    params.time = time;
    params.iter = iter;

    params.region_width = config.region_width;
    params.region_height = config.region_height;
    params.second_viscosity = config.second_viscosity;
    params.dynamic_viscosity = config.dynamic_viscosity;
    params.temperature = config.temperature;
    params.default_density = config.default_density;

    params.dt = config.dt;
    params.max_dt = config.max_dt;
    params.min_dt = config.min_dt;

    params.do_debug = true;
    params.do_stats = true;
    params.do_prints = true;
    return params;
}

void simulation_state_deinit(App_State* app) 
{
    sim_const_state_deinit(&app->gpu_const_state);
    sim_mut_state_deinit(&app->gpu_mut_states[0]);
    sim_mut_state_deinit(&app->gpu_mut_states[1]);
    cpu_sim_const_state_deinit(&app->cpu_const_state);
    cpu_sim_mut_state_deinit(&app->cpu_mut_state);
}

void simulation_state_init(App_State* app, Sim_Config config)
{
    app->config = std::move(config);
    app->cpu_const_state = cpu_sim_const_state_init(app->config.nx, app->config.ny);
    app->cpu_mut_state = cpu_sim_mut_state_init(app->config.nx, app->config.ny);

    sim_const_state_init(&app->gpu_const_state, app->config.nx, app->config.ny);
    sim_mut_state_init(&app->gpu_mut_states[0], app->config.nx, app->config.ny);
    sim_mut_state_init(&app->gpu_mut_states[1], app->config.nx, app->config.ny);

    cudaGetErrorString(cudaSuccess);
}

#ifdef COMPILE_GRAPHICS
#include "gl.h"
#include <GLFW/glfw3.h>

void glfw_resize_func(GLFWwindow* window, int width, int heigth);
void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods);
#endif

enum {
    SAVE_BIN = 4,
    SAVE_STATS = 8,
    SAVE_CONFIG = 16,
    SAVE_ALL = 31
};
bool save_state(App_State* app, int flags, int snapshot_index);
std::string format_save_path(App_State* app, std::string_view format, std::string_view formatname, const char* formatext);

double lerp(double begin, double end, double t)
{
    return (1 - t)*begin + t*end;
}

int main(int argc, char** argv)
{
    int main_out = 0;
    int config_count = argc > 1 ? argc - 1 : 1;
    for(int app_run = 0; app_run < config_count; app_run += 1)
    {
        const char* config_path = "config.ini";
        if(argc > 1)
            config_path = argv[app_run+1];

        App_State app_data = {};
        Sim_Config config = {};

        if(sim_read_config(config_path, &config, NULL, 0) == false)
        {
            LOG_ERROR("app", "failed to read config '%s'. Skipping to next config.", config_path);
            main_out = 1;
            continue;
        }

        main_out = 0;
        if(config.app_run_tests)
            sim_run_tests();
        if(config.app_run_simulation == false)
            continue;

        App_State* app = (App_State*) &app_data;
        app->is_in_step_mode = true;
        app->remaining_steps = 0;
        app->step_by = 1;
        app->init_time = time(NULL);

        simulation_state_init(app, config);

        static File_Logger file_logger = {0};
        {
            std::string logs_path = format_save_path(app, app->config.save_logs_path, "log", "log");
            std::filesystem::create_directories(logs_path);
            file_logger_init(&file_logger, logs_path.data(), 0);
            log_system_set_logger(&file_logger.logger);
        }

        if(config.snapshot_initial_conditions)
            save_state(app, SAVE_BIN | SAVE_CONFIG, 0);

        #define LOG_INFO_CONFIG_FLOAT(var)  LOG_INFO("config", #var " = %.2lf", (double) config.var);
        #define LOG_INFO_CONFIG_INT(var)    LOG_INFO("config", #var " = %lli", (long long) config.var);
        #define LOG_INFO_CONFIG_STRING(var) LOG_INFO("config", #var " = %s", (long long) config.var.data());
        #define LOG_INFO_CONFIG_BOOL(var)   LOG_INFO("config", #var " = %s", config.var ? "true" : "false");

        LOG_INFO_CONFIG_FLOAT(region_width);
        LOG_INFO_CONFIG_FLOAT(region_height);
        LOG_INFO_CONFIG_FLOAT(second_viscosity);
        LOG_INFO_CONFIG_FLOAT(dynamic_viscosity);
        LOG_INFO_CONFIG_FLOAT(temperature);
        LOG_INFO_CONFIG_FLOAT(default_density);

        LOG_INFO_CONFIG_INT(nx);
        LOG_INFO_CONFIG_INT(ny);

        LOG_INFO_CONFIG_FLOAT(dt);
        LOG_INFO_CONFIG_FLOAT(min_dt);
        LOG_INFO_CONFIG_FLOAT(max_dt);

        LOG_INFO_CONFIG_FLOAT(init_ux);
        LOG_INFO_CONFIG_FLOAT(init_uy);
        LOG_INFO_CONFIG_FLOAT(init_rho);

        LOG_INFO_CONFIG_FLOAT(obstacle_center_x);
        LOG_INFO_CONFIG_FLOAT(obstacle_center_y);
        LOG_INFO_CONFIG_FLOAT(obstacle_radius);

        LOG_INFO_CONFIG_FLOAT(intake_t0_ux);
        LOG_INFO_CONFIG_FLOAT(intake_t0_uy);
        LOG_INFO_CONFIG_FLOAT(intake_t0_rho);

        LOG_INFO_CONFIG_FLOAT(intake_tf_ux);
        LOG_INFO_CONFIG_FLOAT(intake_tf_uy);
        LOG_INFO_CONFIG_FLOAT(intake_tf_rho);

        LOG_INFO_CONFIG_FLOAT(intake_t1_ux);
        LOG_INFO_CONFIG_FLOAT(intake_t1_uy);
        LOG_INFO_CONFIG_FLOAT(intake_t1_rho);

        LOG_INFO_CONFIG_FLOAT(simul_t0_t);
        LOG_INFO_CONFIG_FLOAT(simul_tf_t);
        LOG_INFO_CONFIG_FLOAT(simul_t1_t);

        //snapshots
        LOG_INFO_CONFIG_FLOAT(snapshot_every);
        LOG_INFO_CONFIG_INT(snapshot_times);
        LOG_INFO_CONFIG_BOOL(snapshot_initial_conditions);

        LOG_INFO_CONFIG_STRING(runname);
        LOG_INFO_CONFIG_STRING(save_snapshot_path);
        LOG_INFO_CONFIG_STRING(save_stats_path);
        LOG_INFO_CONFIG_STRING(save_config_path);

        //App settings
        LOG_INFO_CONFIG_BOOL(app_run_simulation);
        LOG_INFO_CONFIG_BOOL(app_run_tests);
        LOG_INFO_CONFIG_BOOL(app_run_benchmarks);
        LOG_INFO_CONFIG_BOOL(app_collect_stats);

        LOG_INFO_CONFIG_BOOL(app_interactive_mode);
        LOG_INFO_CONFIG_BOOL(app_linear_filtering);
        LOG_INFO_CONFIG_FLOAT(app_display_min);
        LOG_INFO_CONFIG_FLOAT(app_display_max);

        #undef LOG_INFO_CONFIG_FLOAT
        #undef LOG_INFO_CONFIG_INT
        #undef LOG_INFO_CONFIG_BOOL
        #undef LOG_INFO_CONFIG_STRING

        #ifndef COMPILE_GRAPHICS
        config.interactive_mode = false;
        #endif // COMPILE_GRAPHICS

        //OPENGL setup
        TEST(glfwInit(), "Failed to init glfw");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);  
        glfwWindowHint(GLFW_SAMPLES, 4);  

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        ASSERT(monitor && mode);
        if(monitor != NULL && mode != NULL)
        {
            glfwWindowHint(GLFW_RED_BITS, mode->redBits);
            glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
            glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        }

        GLFWwindow* window = glfwCreateWindow(DEF_WINDOW_WIDTH, DEF_WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
        TEST(window != NULL, "Failed to make glfw window");

        //glfwSetWindowUserPointer(window, &app);
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, app);
        glfwSetFramebufferSizeCallback(window, glfw_resize_func);
        glfwSetKeyCallback(window, glfw_key_func);
        glfwSwapInterval(0);
        gl_init((void*) glfwGetProcAddress);
    
        double time_display_last_time = 0;
        double render_last_time = 0;
        double simulated_last_time = 0;
        double poll_last_time = 0;

        double processing_time = 0;
        double acumulated_processing_time = 0;

        int snapshot_every_i = 0;
        int snapshot_times_i = 0;
        bool end_reached = false;
        bool save_this_iter = false;

        sim_set_constant_initial_conditions(&app->cpu_mut_state, app->config.init_rho, app->config.init_ux, app->config.init_uy, app->config);
        sim_set_boundary_conditions_tube(&app->cpu_const_state, app->config.intake_t0_ux, app->config.intake_t0_ux, app->config.intake_t0_rho, app->config.intake_t0_rho, app->config);
        {
            size_t bytes = (size_t) app->config.nx * (size_t) app->config.ny * (size_t) sizeof(Real);
            sim_modify(app->gpu_mut_states[1].rho, app->cpu_mut_state.rho, bytes, MODIFY_UPLOAD);
            sim_modify(app->gpu_mut_states[1].ux, app->cpu_mut_state.ux, bytes, MODIFY_UPLOAD);
            sim_modify(app->gpu_mut_states[1].uy, app->cpu_mut_state.uy, bytes, MODIFY_UPLOAD);
        }

        double last_intake_ux = 0;
        double last_intake_uy = 0;
        double last_intake_rho = 0;

        double start_time = clock_s();
        while (!glfwWindowShouldClose(window))
        {
            double frame_start_time = clock_s();

            bool update_screen = frame_start_time - render_last_time > SCREEN_UPDATE_PERIOD;
            bool update_frame_time_display = frame_start_time - time_display_last_time > FPS_DISPLAY_PERIOD;
            bool poll_events = frame_start_time - poll_last_time > POLL_EVENTS_PERIOD;

            Sim_Mut_State curr_state = app->gpu_mut_states[app->iter % 2];
            Sim_Mut_State prev_state = app->gpu_mut_states[(app->iter + 1) % 2];

            double next_snapshot_every = (double) (snapshot_every_i + 1) * config.snapshot_every;
            double next_snapshot_times = (double) (snapshot_times_i + 1) * config.simul_t1_t / config.snapshot_times;

            //enforce boundary conditions
            double intake_ux = last_intake_ux;
            double intake_uy = last_intake_uy;
            double intake_rho = last_intake_rho;
            if(app->config.simul_t0_t <= app->sim_time && app->sim_time < app->config.simul_tf_t)
            {
                double t = (app->sim_time - app->config.simul_t0_t)/(app->config.simul_tf_t - app->config.simul_t0_t);
                intake_rho = lerp(app->config.intake_t0_rho, app->config.intake_tf_rho, t);
                intake_ux = lerp(app->config.intake_t0_ux, app->config.intake_tf_ux, t);
                intake_uy = lerp(app->config.intake_t0_uy, app->config.intake_tf_uy, t);
            }
            if(app->config.simul_tf_t <= app->sim_time && app->sim_time < app->config.simul_t1_t)
            {
                double t = (app->sim_time - app->config.simul_tf_t)/(app->config.simul_t1_t - app->config.simul_tf_t);
                intake_rho = lerp(app->config.intake_t1_rho, app->config.intake_t1_rho, t);
                intake_ux = lerp(app->config.intake_t1_ux, app->config.intake_t1_ux, t);
                intake_uy = lerp(app->config.intake_t1_uy, app->config.intake_t1_uy, t);
            }

            if(intake_ux != last_intake_ux || intake_uy != last_intake_uy || intake_rho != last_intake_rho) 
            {
                sim_set_boundary_conditions_tube(&app->cpu_const_state, intake_ux, intake_ux, intake_rho, intake_rho, app->config);

                size_t bytes = (size_t) app->config.nx * (size_t) app->config.ny * (size_t) sizeof(Real);
                size_t bytes_flags = (size_t) app->config.nx * (size_t) app->config.ny * (size_t) sizeof(Sim_Flags);
                
                sim_modify(app->gpu_const_state.flags, app->cpu_const_state.flags, bytes_flags, MODIFY_UPLOAD);

                sim_modify(app->gpu_const_state.set_rho, app->cpu_const_state.set_rho, bytes, MODIFY_UPLOAD);
                sim_modify(app->gpu_const_state.set_ux, app->cpu_const_state.set_ux, bytes, MODIFY_UPLOAD);
                sim_modify(app->gpu_const_state.set_uy, app->cpu_const_state.set_uy, bytes, MODIFY_UPLOAD);

                sim_modify(app->gpu_const_state.set_dx_rho, app->cpu_const_state.set_dx_rho, bytes, MODIFY_UPLOAD);
                sim_modify(app->gpu_const_state.set_dx_ux, app->cpu_const_state.set_dx_ux, bytes, MODIFY_UPLOAD);
                sim_modify(app->gpu_const_state.set_dx_uy, app->cpu_const_state.set_dx_uy, bytes, MODIFY_UPLOAD);
                
                sim_modify(app->gpu_const_state.set_dy_rho, app->cpu_const_state.set_dy_rho, bytes, MODIFY_UPLOAD);
                sim_modify(app->gpu_const_state.set_dy_ux, app->cpu_const_state.set_dy_ux, bytes, MODIFY_UPLOAD);
                sim_modify(app->gpu_const_state.set_dy_uy, app->cpu_const_state.set_dy_uy, bytes, MODIFY_UPLOAD);
            }

            last_intake_ux = intake_ux;
            last_intake_uy = intake_uy;
            last_intake_rho = intake_rho;

            //
            if(app->sim_time >= next_snapshot_every)
            {
                snapshot_every_i += 1;
                save_this_iter = true;
            }

            if(app->sim_time >= next_snapshot_times && end_reached == false)
            {
                snapshot_times_i += 1;
                save_this_iter = true;
            }

            if(config.simul_t1_t - app->sim_time < 1e-16 && end_reached == false)
            {
                LOG_INFO("app", "reached stop time %lfs. Took raw processing %lfs total (with puases) %lfs. Simulation paused.", config.simul_t1_t, acumulated_processing_time, clock_s() - start_time);
                app->is_in_step_mode = true;
                end_reached = true;
                save_this_iter = true;
            }

            if(save_this_iter)
                save_state(app, SAVE_BIN | SAVE_CONFIG | SAVE_STATS, ++app->count_written_snapshots);

            if(update_screen)
            {
                render_last_time = frame_start_time;
                draw_sci_cuda_memory("main", prev_state.nx, prev_state.ny, (float) app->config.app_display_min, (float) app->config.app_display_max, config.app_linear_filtering, prev_state.rho);
                
                Draw_Lines_Config lines_config = {0};
                lines_config.nx = prev_state.nx;
                lines_config.ny = prev_state.ny;
                lines_config.pix_size = 1;

                lines_config.max_size = (float) lines_config.pix_size/prev_state.nx*2;
                lines_config.min_size = 0;
                lines_config.rgba_i0 = 0;
                lines_config.rgba_i1 = 0;

                lines_config.width_i0 = (float) lines_config.pix_size/prev_state.nx/4;
                lines_config.width_i1 = 0.000f;
                lines_config.scale = 0.1f;
                lines_config.dx = 1.0f/prev_state.nx;
                lines_config.dy = 1.0f/prev_state.ny;

                draw_flow_arrows("flow", prev_state.ux, prev_state.uy, lines_config);
                glfwSwapBuffers(window);
            }

            if(update_frame_time_display)
            {
                glfwSetWindowTitle(window, format_string("%s step: %3.3lfms | real: %8.6lfms", "TODO", processing_time * 1000, app->sim_time*1000).c_str());
                time_display_last_time = frame_start_time;
            }

            bool step_sym = false;
            if(app->is_in_step_mode)
                step_sym = app->remaining_steps > 0.5;
            else
                step_sym = frame_start_time - simulated_last_time > FREE_RUN_PERIOD/app->step_by;

            if(step_sym)
            {
                double solver_start_time = clock_s();
                Sim_Params params = sim_params_from_config(app->sim_time, app->iter, app->config);
                app->sim_time += sim_step(&curr_state, &prev_state, &app->gpu_const_state, params);
                simulated_last_time = frame_start_time;
                app->remaining_steps -= 1;
                double solver_end_time = clock_s();

                processing_time = solver_end_time - solver_start_time;
                acumulated_processing_time += processing_time;
                app->iter += 1;
            }

            if(poll_events)
            {
                poll_last_time = frame_start_time;
                glfwPollEvents();
            }

            double end_frame_time = clock_s();
            app->real_dt = end_frame_time - frame_start_time;

            //if is idle for the last 0.5 seconds limit the framerate to IDLE_RENDER_FREQ
            bool do_frame_limiting = simulated_last_time + 0.5 < frame_start_time;
            do_frame_limiting = true;
            if(do_frame_limiting && app->real_dt < SCREEN_UPDATE_IDLE_PERIOD)
                wait_s(SCREEN_UPDATE_IDLE_PERIOD - app->real_dt);

            save_this_iter = false;
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    return main_out;    
}

#ifdef COMPILE_GRAPHICS

void glfw_resize_func(GLFWwindow* window, int width, int heigth)
{
    (void) window;
	glViewport(0, 0, width, heigth);
}

void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void) mods;
    (void) scancode;
    (void) window;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;

    if(action == GLFW_RELEASE)
    {
        if(key == GLFW_KEY_ENTER)
            app->remaining_steps = app->step_by;
        if(key == GLFW_KEY_SPACE)
        {
            app->is_in_step_mode = !app->is_in_step_mode;
            LOG_INFO("APP", "Simulation %s", app->is_in_step_mode ? "paused" : "running");
        }
        if(key == GLFW_KEY_D)
        {
            app->is_in_debug_mode = !app->is_in_debug_mode;
            LOG_INFO("APP", "Debug %s", app->is_in_debug_mode ? "true" : "false");
        }
        if(key == GLFW_KEY_S)
        {
            LOG_INFO("APP", "On demand snapshot triggered");
            save_state(app, SAVE_BIN | SAVE_CONFIG | SAVE_STATS, ++app->count_written_snapshots);
        }
        if(key == GLFW_KEY_R)
        {
            LOG_INFO("APP", "Input range to display in form 'MIN space MAX'");

            double new_display_max = app->config.app_display_max;
            double new_display_min = app->config.app_display_min;
            if(scanf("%lf %lf", &new_display_min, &new_display_max) != 2)
            {
                LOG_INFO("APP", "Bad range syntax!");
            }
            else
            {
                LOG_INFO("APP", "displaying range [%.2lf, %.2lf]", new_display_min, new_display_max);
                app->config.app_display_max = (Real) new_display_max;
                app->config.app_display_min = (Real) new_display_min;
            }
        }

        if(key == GLFW_KEY_P)
        {
            LOG_INFO("APP", "Input simulation speed modifier in form 'NUM'");
            double new_step_by = app->step_by;
            if(scanf("%lf", &new_step_by) != 1)
            {
                LOG_INFO("APP", "Bad speed syntax!");
            }
            else
            {
                LOG_INFO("APP", "using simulation speed %.2lf", new_step_by);
                app->step_by = new_step_by;
            }
        }

        int new_render_target = -1;
        if(key == GLFW_KEY_F1) new_render_target = 0;
        if(key == GLFW_KEY_F2) new_render_target = 1;
        if(key == GLFW_KEY_F3) new_render_target = 2;
        if(key == GLFW_KEY_F4) new_render_target = 3;
        if(key == GLFW_KEY_F5) new_render_target = 4;
        if(key == GLFW_KEY_F6) new_render_target = 5;
        if(key == GLFW_KEY_F7) new_render_target = 6;
        if(key == GLFW_KEY_F8) new_render_target = 7;
        
        if(new_render_target != -1)
        {
            LOG_INFO("APP", "redering %s", "TODO");
            app->render_target = new_render_target;
        }
    }
}
#endif

#include <chrono>
static double clock_s()
{
    static int64_t init_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double unit = (double) std::chrono::high_resolution_clock::period::den;
    double clock = (double) (now - init_time) / unit;
    return clock;
}

#include <thread>
static void wait_s(double seconds)
{
    auto now = std::chrono::high_resolution_clock::now();
    auto sleep_til = now + std::chrono::microseconds((int64_t) (seconds * 1000*1000));
    std::this_thread::sleep_until(sleep_til);
}
#include <cstdio>

#define MAP_FILE_MAGIC "crystmap"
#define MAP_TYPE_CHAR "char"
#define MAP_TYPE_I16 "i16"
#define MAP_TYPE_I32 "i32"
#define MAP_TYPE_F32 "f32"
#define MAP_TYPE_F64 "f64"

typedef struct Bin_Map_File {
    int64_t nx;
    int64_t ny;
    double width;
    double height;
    double time;
    int64_t iter;
    int64_t type_size;
    char name[64];
    char type[64];
    char meta[64];
    void* data;
} Bin_Map_File;

bool save_bin_map_file(const char* filename, const char* write_mode, const Bin_Map_File* maps, isize maps_count)
{
    bool state = false;
    FILE* file = fopen(filename, write_mode);
    if(file != NULL)
    {
        for(isize i = 0; i < maps_count; i++) {
            size_t N = (size_t) maps[i].nx * (size_t) maps[i].ny;
            fwrite(MAP_FILE_MAGIC, 8, 1, file);
            fwrite(&maps[i], sizeof maps[i], 1, file);
            fwrite(maps[i].data, (size_t) maps[i].type_size, N, file);
        }

        state = ferror(file) == 0;
        fclose(file);
    }
    if(state == false)
        LOG_ERROR("APP", "Error saving bin file '%s'", filename);
    return state;
}

struct Small_String {
    char data[16];
};

template<typename T>
Small_String float_vec_string_at(const std::vector<T>& arr, size_t at)
{
    Small_String out = {0};
    if(at < arr.size())
        snprintf(out.data, sizeof out.data, "%f", (float) arr[at]);
    return out;
}

template<typename T>
Small_String int_vec_string_at(const std::vector<T>& arr, size_t at)
{
    Small_String out = {0};
    if(at < arr.size())
        snprintf(out.data, sizeof out.data, "%lli", (long long) arr[at]);
    return out;
}

#include <filesystem>
std::string format_save_path(App_State* app, std::string_view format, std::string_view formatname, const char* formatext)
{
    time_t _curr_time = time(NULL);
    tm* s = localtime(&app->init_time);
    tm* c = localtime(&_curr_time);

    //can include %{startdate} %{starttime} %{date} %{time} %{saveindex} %{iter} %{simtime} %{realtime} %{runname} %{ext}
    std::string startdate = format_string("%04i-%02i-%02i", s->tm_year + 1900, s->tm_mon, s->tm_mday);
    std::string starttime = format_string("%02i-%02i-%02i", s->tm_hour, s->tm_min, s->tm_sec);
    std::string date = format_string("%04i-%02i-%02i", c->tm_year + 1900, c->tm_mon, c->tm_mday);
    std::string time = format_string("%02i-%02i-%02i", c->tm_hour, c->tm_min, c->tm_sec);
    std::string saveindex = format_string("%02lli", app->save_index);
    std::string iter = format_string("%06lli", app->iter);
    std::string simtime = format_string("%e", app->sim_time);
    std::string realtime = format_string("%e", clock_s());
    std::string const& runname = app->config.runname;

    std::string processed;

    for(size_t i = 0; i < format.size(); i++)
    {
        bool did_match = false;
        if(i + 1 < format.size() && format[i] == '%' && format[i+1] == '{')
        {
            size_t end = format.find('}', i+2);
            if((i == 0 || format[i-1] != '\\') && end != (size_t) -1)
            {
                std::string_view portion = format.substr(i+2, end - (i+2));
                if(0) {}
                else if(portion == "startdate") processed.append(startdate);
                else if(portion == "starttime") processed.append(starttime);
                else if(portion == "date") processed.append(date);
                else if(portion == "time") processed.append(time);
                else if(portion == "saveindex") processed.append(saveindex);
                else if(portion == "iter") processed.append(iter);
                else if(portion == "simtime") processed.append(simtime);
                else if(portion == "realtime") processed.append(realtime);
                else if(portion == "runname") processed.append(runname);
                else if(portion == "formatname") processed.append(formatname);
                else if(portion == "formatext") processed.append(formatext);
                else
                    LOG_ERROR("APP", "bad special replace '%s' in save path format '%s'", std::string(portion).data(), format.data());

                i = end;
                did_match = true;
            }
        }

        if(did_match == false)
            processed.push_back(format[i]);
    }

    return processed;
}

bool save_state(App_State* app, int flags, int snapshot_index)
{
    bool state = true;
    if(flags & SAVE_BIN)    
    {
        std::string snapshot_path = format_save_path(app, app->config.save_snapshot_path, "crystmap", "crystmap");
        std::filesystem::create_directories(snapshot_path);
        LOG_INFO("APP", "TODO: save maps at '%s'", snapshot_path.data());
    }

    if((flags & SAVE_STATS))    
    {
        std::string stats_path = format_save_path(app, app->config.save_stats_path, "stats", "csv");
        std::filesystem::create_directories(stats_path);
        LOG_INFO("APP", "TODO: save maps at '%s'", stats_path.data());
    } 

    if((flags & SAVE_CONFIG) && app->count_written_configs == 0)    
    {
        std::string stats_path = format_save_path(app, app->config.save_config_path, "config", "ini");
        std::filesystem::create_directories(stats_path);
        LOG_INFO("APP", "TODO: save maps at '%s'", stats_path.data());
    } 

    return state;
}
