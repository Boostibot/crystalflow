#pragma once
#include <string>


enum Skip_Options {
    SKIP_NORMAL,
    SKIP_INVERSE,
};

size_t skip_set(size_t from, std::string_view source, std::string_view char_set, Skip_Options options = SKIP_NORMAL)
{
    for(size_t i = from; i < source.size(); i++)
    {
        char c = source[i];
        bool is_in_set = char_set.find(c, 0) != std::string::npos;
        if(is_in_set && options == SKIP_INVERSE)
            return i;
        if(is_in_set == false && options == SKIP_NORMAL)
            return i;
    }

    return source.size();
}

size_t skip_set_reverse(size_t from, size_t to, std::string_view source, std::string_view char_set, Skip_Options options = SKIP_NORMAL)
{
    for(size_t i = from; i-- > to;)
    {
        char c = source[i];
        bool is_in_set = char_set.find(c, 0) != std::string::npos;
        if(is_in_set && options == SKIP_INVERSE)
            return i + 1;
        if(is_in_set == false && options == SKIP_NORMAL)
            return i + 1;
    }

    return to;
}

std::string_view get_line(size_t* positon, std::string_view source)
{
    if(*positon >= source.size())
        return "";

    size_t line_start = *positon;
    size_t line_end = source.find('\n', *positon);
    if(line_end == std::string::npos)
        line_end = source.size();

    std::string_view out = source.substr(line_start, line_end - line_start);
    *positon = line_end + 1;
    return out;
}

#include "log.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

bool file_read_entire(const char* path, std::string* into)
{
    std::ifstream t(path);
    if(t.is_open() == false)
        return false;

    std::stringstream buffer;
    buffer << t.rdbuf();
    if(into)
        *into = buffer.str();

    return t.bad() == false;
}

bool file_write_entire(const char* path, std::string_view content)
{
    std::ofstream t;
    t.open(path);
    t << content;
    return !t.bad();
}

#define WHITESPACE " \t\n\r\v\f"
#define MARKERS    "=:"

enum Key_Value_Type {
    KEY_VALUE_BLANK,
    KEY_VALUE_COMMENT,
    KEY_VALUE_SECTION,
    KEY_VALUE_ENTRY,
    KEY_VALUE_ERROR,
};

Key_Value_Type match_key_value_ini_pair(std::string_view line, std::string* key, std::string* value, std::string* section)
{
    size_t key_from = skip_set(0, line, WHITESPACE);
    size_t key_to = skip_set(key_from, line, WHITESPACE MARKERS, SKIP_INVERSE);
    
    if(key_from == line.size())
        return KEY_VALUE_BLANK;

    size_t marker_from = skip_set(key_to, line, WHITESPACE);
    size_t marker_to = skip_set(marker_from, line, MARKERS);

    size_t value_from = skip_set(marker_to, line, WHITESPACE);
    size_t value_to = skip_set_reverse(line.size(), value_from, line, WHITESPACE);

    std::string_view found_key = line.substr(key_from, key_to - key_from);
    std::string_view found_marker = line.substr(marker_from, marker_to - marker_from);
    std::string_view found_value = line.substr(value_from, value_to - value_from);

    //If is comment
    if(found_key.size() > 0 && (found_key[0] == '#' || found_key[0] == ';'))
        return KEY_VALUE_COMMENT;

    //If is section
    if(found_key.size() > 0 && found_key.front() == '[' && found_key.back() == ']')
    {
        *section = found_key.substr(1, found_key.size() - 2);
        return KEY_VALUE_SECTION;
    }

    if(found_marker == "=" || found_marker == ":" || found_marker == ",")
    {
        //Remove comment from value
        size_t comment_index1 = found_value.find(';');
        size_t comment_index2 = found_value.find('#');
        if(comment_index1 == (size_t) -1)
            comment_index1 = found_value.size();
        if(comment_index2 == (size_t) -1)
            comment_index2 = found_value.size();

        size_t comment_index = comment_index1 < comment_index2 ? comment_index1 : comment_index2;
        size_t comment_removed_value_to = skip_set_reverse(comment_index, 0, found_value, WHITESPACE);

        std::string_view comment_removed_value = found_value.substr(0, comment_removed_value_to);

        *key = found_key;
        *value = comment_removed_value;
        return KEY_VALUE_ENTRY;
    }
    else
    {
        return KEY_VALUE_ERROR;
    }
}

#define SECTION_MARKER "~!~!~!~" //A very unlikely set of characters

using Key_Value = std::unordered_map<std::string, std::string>;

void key_value_ini_append(Key_Value* key_val, std::string_view source)
{
    int line_number = 1;
    std::string key;
    std::string value;
    std::string current_section;
    for(size_t i = 0; i < source.size(); line_number++)
    {
        std::string_view line = get_line(&i, source);
        key.clear();
        value.clear();
        
        Key_Value_Type type = match_key_value_ini_pair(line, &key, &value, &current_section);
        if(type == KEY_VALUE_ERROR)
            LOG_ERROR("to_key_value_ini", "invalid syntax on line %i '%s'. Ignoring.", line_number, std::string(line).c_str());
        else if(type == KEY_VALUE_ENTRY)
            (*key_val)[current_section + SECTION_MARKER + key] = value;
    }
}

std::string key_value_ini_key(const char* section, const char* str)
{
    return std::string(section) + SECTION_MARKER + str;
}

bool key_value_ini_get_any(const Key_Value& map, void* out, const char* section, const char* str, const char* type_fmt, const char* type)
{
    std::string key = key_value_ini_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find %s '%s' in section [%s]", type, str, section);
    else
    {
        const char* val = found->second.c_str();  
        state = sscanf(val, type_fmt, out) == 1;
        if(state == false)
            LOG_ERROR("config", "Couldnt match %s. Got: '%s'. While parsing value '%s'", type, val, str);
    }
    return state;
}

bool key_value_ini_get_vec2(const Key_Value& map, double* out_x, double* out_y, const char* section, const char* str)
{
    std::string key = key_value_ini_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find Vec2 '%s' in section [%s]", str, section);
    else
    {
        const char* val = found->second.c_str();  
        state = sscanf(val, "%lf %lf", out_x, out_y) == 2;
        if(state == false)
            LOG_ERROR("config", "Couldnt match Vec2. Got: '%s'. While parsing value '%s'", val, str);
    }
    return state;
}

bool key_value_ini_get_int64(const Key_Value& map, int64_t* out, const char* section, const char* str)
{
    return key_value_ini_get_any(map, out, section, str, "%lli", "int64_t");
}

bool key_value_ini_get_float(const Key_Value& map, float* out, const char* section, const char* str)
{
    return key_value_ini_get_any(map, out, section, str, "%f", "float");
}

bool key_value_ini_get_double(const Key_Value& map, double* out, const char* section, const char* str)
{
    return key_value_ini_get_any(map, out, section, str, "%lf", "double");
}

bool key_value_ini_get_string(const Key_Value& map, std::string* out, const char* section, const char* str)
{
    std::string key = key_value_ini_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state)
        *out = found->second;
    else
        LOG_ERROR("config", "couldnt find or match string '%s' in section [%s]", str, section);
    return state;
}

bool key_value_ini_get_bool(const Key_Value& map, bool* out, const char* section, const char* str)
{
    std::string key = key_value_ini_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find bool '%s' in section [%s]", str, section);
    else
    {
        if(found->second == "true" || found->second == "1")
            *out = true;
        else if(found->second == "false" || found->second == "0")
            *out = false;
        else
        {
            LOG_ERROR("config", "couldnt match bool '%s = %s' ", str, found->second.c_str());
            state = false;
        }
    }

    return state;
}

#include <stdarg.h>
std::string format_string(const char* format, ...)
{
    std::string out;
    va_list args;
    va_start(args, format);

    va_list args_copy;
    va_copy(args_copy, args);

    int size = vsnprintf(NULL, 0, format, args_copy);

    out.resize((size_t) size+5);
    size = vsnprintf(&out[0], out.size(), format, args);
    out.resize((size_t) size);
    va_end(args);

    return out;
}

typedef struct Sim_Config{
    std::string entire_config_file;

    double region_width;
    double region_height;
    double second_viscosity;
    double dynamic_viscosity;
    double temperature;
    double default_density;

    int64_t nx;
    int64_t ny;

    double dt;
    double min_dt;
    double max_dt;

    double init_ux;
    double init_uy;
    double init_rho;

    double obstacle_center_x;
    double obstacle_center_y;
    double obstacle_radius;

    double intake_t0_ux;
    double intake_t0_uy;
    double intake_t0_rho;

    double intake_tf_ux;
    double intake_tf_uy;
    double intake_tf_rho;

    double intake_t1_ux;
    double intake_t1_uy;
    double intake_t1_rho;

    double simul_t0_t;
    double simul_tf_t;
    double simul_t1_t;

    //snapshots
    double snapshot_every;
    int64_t snapshot_times;
    bool snapshot_initial_conditions;

    std::string runname;

    //can include fromat string in the format '%{format_name}' where format name can be one of
    //"startdate" - date of the simulation start in format 2025-03-05
    //"starttime" - time of the simulation start in format 22-53-07
    //"date" - current date in format 2025-03-05
    //"time" - current time in format 22-53-07
    //"saveindex" - index of the specific save that gets incremented on every use
    //"iter" - iteration of the simulation
    //"simtime" - simulation time in scientific notation
    //"realtime" - current time in scientific notation
    //"runname" - value of runname
    //"formatname" - extension of the specific file format
    //"formatext" - extension of the specific file format
    //
    //for example "snapshots/%{runname}%{startdate}__%{starttime}/%{saveindex}_%{iter}_%{formatname}.%{formatext}"
    std::string save_snapshot_path; 
    std::string save_stats_path;
    std::string save_config_path;
    std::string save_logs_path;

    //App settings
    bool app_run_simulation;
    bool app_run_tests;
    bool app_run_benchmarks;
    bool app_collect_stats;

    bool   app_interactive_mode;
    bool   app_linear_filtering;
    double app_display_min;
    double app_display_max;
} Sim_Config;

#include <filesystem>
bool sim_read_config(const char* path, Sim_Config* config, const char* overrides[], int overrides_count)
{
    Sim_Config null_config = {};
    *config = null_config; 

    bool state = file_read_entire(path, &config->entire_config_file);
    if(state == false)
        LOG_ERROR("config", "coudlnt read config file '%s'. Current working directory '%s'", path, std::filesystem::current_path().string().c_str());
    else
    {
        Key_Value pairs = {};
        key_value_ini_append(&pairs, config->entire_config_file);
        for(int i = 0; i < overrides_count; i++)
            key_value_ini_append(&pairs, overrides[i]);

        uint8_t read_state = true;
        
        read_state &= (uint8_t) key_value_ini_get_int64(pairs, &config->nx, "simulation", "nx");
        read_state &= (uint8_t) key_value_ini_get_int64(pairs, &config->ny, "simulation", "ny");;
        double dx = 0;
        double dy = 0;
        if(key_value_ini_get_double(pairs, &config->region_width, "simulation", "region_width")
            && key_value_ini_get_double(pairs, &config->region_height, "simulation", "region_height"))
        {}
        else if(key_value_ini_get_double(pairs, &dx, "simulation", "dx")
            && key_value_ini_get_double(pairs, &dy, "simulation", "dy"))
        {
            config->region_width = dx*config->nx;
            config->region_height = dy*config->ny;
        }
        else
            read_state = false;

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->dt, "simulation", "dt");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->second_viscosity, "simulation", "second_viscosity");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->dynamic_viscosity, "simulation", "dynamic_viscosity");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->temperature, "simulation", "temperature");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->default_density, "simulation", "default_density");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->dt, "simulation", "dt");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->min_dt, "simulation", "min_dt");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->max_dt, "simulation", "max_dt");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->init_ux, "simulation", "init_ux");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->init_uy, "simulation", "init_uy");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->init_rho, "simulation", "init_rho");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->obstacle_center_x, "simulation", "obstacle_center_x");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->obstacle_center_y, "simulation", "obstacle_center_y");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->obstacle_radius, "simulation", "obstacle_radius");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_t0_ux, "simulation", "intake_t0_ux");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_t0_uy, "simulation", "intake_t0_uy");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_t0_rho, "simulation", "intake_t0_rho");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_tf_ux, "simulation", "intake_tf_ux");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_tf_uy, "simulation", "intake_tf_uy");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_tf_rho, "simulation", "intake_tf_rho");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_t1_ux, "simulation", "intake_t1_ux");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_t1_uy, "simulation", "intake_t1_uy");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->intake_t1_rho, "simulation", "intake_t1_rho");

        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->simul_t0_t, "simulation", "simul_t0_t");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->simul_tf_t, "simulation", "simul_tf_t");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->simul_t1_t, "simulation", "simul_t1_t");

            //snapshots
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->snapshot_every, "snapshots", "snapshot_every");
        read_state &= (uint8_t) key_value_ini_get_int64(pairs, &config->snapshot_times, "snapshots", "snapshot_times");
        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->snapshot_initial_conditions, "snapshots", "snapshot_initial_conditions");
        (uint8_t) key_value_ini_get_string(pairs, &config->runname, "snapshots", "runname");
        read_state &= (uint8_t) key_value_ini_get_string(pairs, &config->save_snapshot_path, "snapshots", "save_snapshot_path");
        read_state &= (uint8_t) key_value_ini_get_string(pairs, &config->save_stats_path, "snapshots", "save_stats_path");
        read_state &= (uint8_t) key_value_ini_get_string(pairs, &config->save_config_path, "snapshots", "save_config_path");
        read_state &= (uint8_t) key_value_ini_get_string(pairs, &config->save_logs_path, "snapshots", "save_logs_path");

            //App settings
        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->app_run_simulation, "settings", "app_run_simulation");
        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->app_run_tests, "settings", "app_run_tests");
        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->app_run_benchmarks, "settings", "app_run_benchmarks");
        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->app_collect_stats, "settings", "app_collect_stats");

        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->app_interactive_mode, "settings", "app_interactive_mode");
        read_state &= (uint8_t) key_value_ini_get_bool(pairs, &config->app_linear_filtering, "settings", "app_linear_filtering");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->app_display_min, "settings", "app_display_min");
        read_state &= (uint8_t) key_value_ini_get_double(pairs, &config->app_display_max, "settings", "app_display_max");
            
        state = read_state;
        if(state == false)
            LOG_ERROR("config", "couldnt find or parse some config entries. Config is only partially loaded!");
        else
            LOG_OKAY("config", "config successfully read!");
    }

    return state;
}
