#pragma once

#include "log.h"
#include "assert.h"

void gl_init(void* load_function);
void draw_colormap(int width, int height, float min, float max, bool linear_filtering, const Real* cuda_memory, const Sim_Flags* flags_or_null);

#if 1

#define GLAD_GL_IMPLEMENTATION
#include "external/glad/glad.h"

const char* gl_translate_error(GLenum code)
{
    switch (code)
    {
        case GL_INVALID_ENUM:                  return "INVALID_ENUM";
        case GL_INVALID_VALUE:                 return "INVALID_VALUE";
        case GL_INVALID_OPERATION:             return "INVALID_OPERATION";
        case GL_STACK_OVERFLOW:                return "STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW:               return "STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY:                 return "OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "INVALID_FRAMEBUFFER_OPERATION";
        default:                               return "UNKNOWN_ERROR";
    }
}

GLenum _gl_check_error(const char *file, int line)
{
    GLenum errorCode = 0;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        const char* error = gl_translate_error(errorCode);
        LOG_ERROR("opengl", "GL error %s | %s (%d)", error, file, line);
    }
    return errorCode;
}

#define gl_check_error() _gl_check_error(__FILE__, __LINE__) 

void gl_debug_output_func(GLenum source, 
                            GLenum type, 
                            unsigned int id, 
                            GLenum severity, 
                            GLsizei length, 
                            const char *message, 
                            const void *userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    (void) length;
    (void) userParam;
    
    Log_Type log_type = LOG_INFO;
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         log_type = LOG_FATAL; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       log_type = LOG_ERROR; break;
        case GL_DEBUG_SEVERITY_LOW:          log_type = LOG_WARN;  break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: log_type = LOG_INFO;  break;
    };

    LOG("opengl", log_type, "GL error (%d): %s", (int) id, message);

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             LOG(">opengl", log_type, "Source: API"); break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   LOG(">opengl", log_type, "Source: Window System"); break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: LOG(">opengl", log_type, "Source: Shader Compiler"); break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     LOG(">opengl", log_type, "Source: Third Party"); break;
        case GL_DEBUG_SOURCE_APPLICATION:     LOG(">opengl", log_type, "Source: Application"); break;
        case GL_DEBUG_SOURCE_OTHER:           LOG(">opengl", log_type, "Source: Other"); break;
    };

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               LOG(">opengl", log_type, "Type: Error"); break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: LOG(">opengl", log_type, "Type: Deprecated Behaviour"); break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  LOG(">opengl", log_type, "Type: Undefined Behaviour"); break; 
        case GL_DEBUG_TYPE_PORTABILITY:         LOG(">opengl", log_type, "Type: Portability"); break;
        case GL_DEBUG_TYPE_PERFORMANCE:         LOG(">opengl", log_type, "Type: Performance"); break;
        case GL_DEBUG_TYPE_MARKER:              LOG(">opengl", log_type, "Type: Marker"); break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          LOG(">opengl", log_type, "Type: Push Group"); break;
        case GL_DEBUG_TYPE_POP_GROUP:           LOG(">opengl", log_type, "Type: Pop Group"); break;
        case GL_DEBUG_TYPE_OTHER:               LOG(">opengl", log_type, "Type: Other"); break;
    };
}


static void gl_post_call_gl_callback(void *ret, const char *name, GLADapiproc apiproc, int len_args, ...) {
    GLenum error_code;

    (void) ret;
    (void) apiproc;
    (void) len_args;

    error_code = glad_glGetError();

    if (error_code != GL_NO_ERROR) 
        LOG_ERROR("opengl", "error %s in %s!", gl_translate_error(error_code), name);
}

void gl_debug_output_enable()
{
    int flags = 0; 
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        LOG_INFO("opengl", "Debug info enabled");
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); 
        glDebugMessageCallback(gl_debug_output_func, NULL);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    } 

    gladSetGLPostCallback(gl_post_call_gl_callback);
    gladInstallGLDebug();
}

unsigned compile_shader(const char* vertex_shader_source, const char* frag_shader_source)
{
    unsigned vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertexShader);

    unsigned fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &frag_shader_source, NULL);
    glCompileShader(fragmentShader);

    unsigned shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    int vertex_success = false; 
    int fragment_success = false;
    int link_success = false;
    char error_msg[512] = {0};

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertex_success);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragment_success);
    if(!vertex_success)
    {
        glGetShaderInfoLog(vertexShader, sizeof error_msg, NULL, error_msg);
        LOG_ERROR("opengl", "Error compiling vertex shader:");
        LOG_ERROR(">opengl", "%s", error_msg);
    }
       
    if(!fragment_success)
    {
        glGetShaderInfoLog(fragmentShader, sizeof error_msg, NULL, error_msg);
        LOG_ERROR("opengl", "Error compiling fragment shader:");
        LOG_ERROR(">opengl", "%s", error_msg);
    }

    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &link_success);
    if(!link_success)
    {
        glGetProgramInfoLog(shaderProgram, sizeof error_msg, NULL, error_msg);
        LOG_ERROR("opengl", "Error linkin shader program:");
        LOG_ERROR(">opengl", "%s", error_msg);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if(!vertex_success || !fragment_success || !link_success)
    {
        glDeleteProgram(shaderProgram);
        return 0;
    }
    else
        return shaderProgram;
}

void draw_screen_quad()
{
    static unsigned quadVAO = 0;
    static unsigned quadVBO = 0;
	if (quadVAO == 0)
	{
		float quadVertices[] = {
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void draw_gl_colormap(unsigned texture, unsigned flags_map, uint32_t flagged_mask, uint32_t flagged_color, float min, float max, float view[4][4], float model[4][4])
{
    enum { MIN_LOCATION = 1 };
    enum { MAX_LOCATION = 2 };
    enum { TEX_LOCATION = 3 };
    enum { FLAGS_LOCATION = 4 };
    enum { FLAGS_MASK_LOCATION = 5 };
    enum { FLAGS_COLOR_LOCATION = 6 };

    enum { TEXTURE_BINDING = 0 };
    enum { FLAGS_BINDING = 1 };

    static bool shader_error = false;
    static unsigned sci_shader = 0;
    if(sci_shader == 0 && shader_error == false)
    {
        const char* frag_shader_source = R"SHADER(
            #version 430 core
            #extension GL_ARB_explicit_uniform_location : require

            layout (location = 1) uniform float _min; //MIN_LOCATION
            layout (location = 2) uniform float _max; //MAX_LOCATION
            layout (location = 3) uniform sampler2D tex; //TEX_LOCATION
            layout (location = 4) uniform usampler2D flags; //FLAGS_LOCATION
            layout (location = 5) uniform uint flags_mask; //FLAGS_MASK_LOCATION
            layout (location = 6) uniform uint flags_color; //FLAGS_MASK_LOCATION

            out vec4 color;
            in vec2 uv;

            #define PI 3.14159265359

            void main()
            {
                float minVal = _min;
                float maxVal = _max;
                vec2 reverse_uv = vec2(uv.x, uv.y);

                if(flags_mask != 0 && (int(texture(flags, reverse_uv).r) & int(flags_mask)) != 0) {
                    int r = (int(flags_color) >> 16) & int(0xFF);
                    int g = (int(flags_color) >> 8) & int(0xFF);
                    int b = (int(flags_color) >> 0) & int(0xFF);
                    int a = 255 - (int(flags_color) >> 24) & int(0xFF);
                    color = vec4(r/255.0, g/255.0, b/255.0, a/255.0);
                    return;
                }

                vec3 texCol = texture(tex, reverse_uv).rgb;      
                float val = texCol.r;
                if(isnan(val))
                {
                    color = vec4(1, 0, 1, 1); //Bright purple
                }
                else if(val < minVal)
                {
                    //Shades from dark gray to black
                    float display = (1 - atan(minVal - val)/PI*2)*0.3;
                    color = vec4(display, display, display, 1.0);
                }
                else if(val > maxVal)
                {
                    //Shades from bright gray to white
                    float display = (atan(val - minVal)/PI*2*0.3 + 0.7);
                    color = vec4(display, display, display, 1.0);
                }
                else
                {
                    //Spectrum blue -> cyan -> green -> yellow -> red

                    val = min(max(val, minVal), maxVal- 0.0001);
                    float d = maxVal - minVal;
                    val = d == 0.0 ? 0.5 : (val - minVal) / d;
                    float m = 0.25;
                    float num = floor(val / m);
                    float s = (val - num * m) / m;
                    float r = 0, g = 0, b = 0;

                    switch (int(num)) {
                        case 0 : r = 0.0; g = s; b = 1.0; break;
                        case 1 : r = 0.0; g = 1.0; b = 1.0-s; break;
                        case 2 : r = s; g = 1.0; b = 0.0; break;
                        case 3 : r = 1.0; g = 1.0 - s; b = 0.0; break;
                    }
                    
                    color = vec4(r, g, b, 1.0);
                }
            }
        )SHADER";

        const char* vertex_shader_source = R"SHADER(
            #version 430 core

            layout (location = 0) in vec3 a_pos;
            layout (location = 1) in vec2 a_uv;

            out vec2 uv;

            void main()
            {
                uv = a_uv;
                gl_Position = vec4(a_pos, 1.0);
            }
        )SHADER";

        sci_shader = compile_shader(vertex_shader_source, frag_shader_source);
        shader_error = sci_shader == 0;
    }
    
    if(shader_error == false)
    {
	    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(sci_shader);
        glBindTexture(GL_TEXTURE_2D, texture);
        glActiveTexture(GL_TEXTURE0 + TEXTURE_BINDING);
        glUniform1i(TEX_LOCATION, TEXTURE_BINDING);

        if(flagged_mask) {
            glBindTexture(GL_TEXTURE_2D, flags_map);
            glActiveTexture(GL_TEXTURE0 + FLAGS_BINDING);
            glUniform1i(FLAGS_LOCATION, FLAGS_BINDING);
            glUniform1ui(FLAGS_COLOR_LOCATION, flagged_color);
        }
    
        glUniform1f(MIN_LOCATION, min);
        glUniform1f(MAX_LOCATION, max);
        glUniform1ui(FLAGS_MASK_LOCATION, flagged_mask);

	    draw_screen_quad();
    }
}

#include "cuda_util.cuh"
void draw_vertices(Sim_Color_Vertex* cuda_vertices, isize count, float view[4][4], float model[4][4])
{
    enum {MAX_VERTICES = 1024*512*3};
    
    static GLuint VBO = 0;
    static GLuint VAO = 0;
    static GLuint shader = 0;
    if(VBO == 0 || VAO == 0) {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, MAX_VERTICES*sizeof(Sim_Color_Vertex), NULL, GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Sim_Color_Vertex), (void*) offsetof(Sim_Color_Vertex, x));
        glVertexAttribIPointer(1, 1, GL_INT, sizeof(Sim_Color_Vertex), (void*) offsetof(Sim_Color_Vertex, packed_color));
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glBindVertexArray(0); 

        const char* frag_shader_source = R"SHADER(
            #version 330 core
            out vec4 out_color;
            in vec4 v_color;

            void main()
            {
                out_color = v_color;
            }
        )SHADER";

        const char* vertex_shader_source = R"SHADER(
            #version 330 core

            layout (location = 0) in vec2 a_pos;
            layout (location = 1) in int a_color;

            out vec4 v_color;

            void main()
            {
                int r = (a_color >> 16) & int(0xFF);
                int g = (a_color >> 8) & int(0xFF);
                int b = (a_color >> 0) & int(0xFF);
                int a = 255 - ((a_color >> 24) & int(0xFF));
                a = 255;
                v_color = vec4(r/255.0, g/255.0, b/255.0, a/255.0);
                gl_Position = vec4(a_pos, 1, 1);
            }
        )SHADER";

        shader = compile_shader(vertex_shader_source, frag_shader_source);
    }

    Sim_Color_Vertex* cpu_vertices = (Sim_Color_Vertex*) malloc((size_t) count*sizeof(Sim_Color_Vertex));
    cudaMemcpy(cpu_vertices, cuda_vertices, (size_t) count*sizeof(Sim_Color_Vertex), cudaMemcpyDeviceToHost);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(VAO);
    glUseProgram(shader); 
    for(isize i = 0; i < count; i += MAX_VERTICES)
    {
        isize copy_size = MIN(count - i, MAX_VERTICES);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLuint) copy_size*sizeof(Sim_Color_Vertex), cpu_vertices);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei) copy_size);
    }
    free(cpu_vertices);
}

static Sim_Color_Vertex* g_cuda_vertices = NULL;
static isize g_cuda_vertices_count = 1024*1024;

void draw_flow_arrows(Real* cuda_uxs, Real* cuda_uys, Draw_Lines_Config config)
{
    if(g_cuda_vertices == NULL) 
        CUDA_TEST(cudaMalloc(&g_cuda_vertices, (size_t) g_cuda_vertices_count*6*sizeof(Sim_Color_Vertex)));

    //TODO: per batch operation
    isize count = 0;
    sim_make_flow_vertices(g_cuda_vertices, &count, g_cuda_vertices_count, cuda_uxs, cuda_uys, config);
    draw_vertices(g_cuda_vertices, count, NULL, NULL);
}

void draw_face_values(Sim_Face_Values* faces_x, Sim_Face_Values* faces_y, isize member_offset, isize nx, isize ny, float width, float min_value, float max_value)
{
    if(g_cuda_vertices == NULL) 
        CUDA_TEST(cudaMalloc(&g_cuda_vertices, (size_t) g_cuda_vertices_count*6*sizeof(Sim_Color_Vertex)));

    Strided_2D_Span faces_x_span = {0};
    faces_x_span.data = (uint8_t*) faces_x + member_offset;
    faces_x_span.nx = nx + 1;
    faces_x_span.ny = ny;
    faces_x_span.stride = sizeof *faces_x;
    faces_x_span.pitch = faces_x_span.nx*(isize) sizeof *faces_x;

    Strided_2D_Span faces_y_span = {0};
    faces_y_span.data = (uint8_t*) faces_y + member_offset;
    faces_y_span.nx = nx;
    faces_y_span.ny = ny + 1;
    faces_y_span.stride = sizeof *faces_y;
    faces_y_span.pitch = faces_y_span.nx*(isize) sizeof *faces_y;

    Draw_Walls_Params walls_config_x = {0};
    walls_config_x.is_x_dir = true;
    walls_config_x.dx = 1.0f/nx;
    walls_config_x.dy = 1.0f/ny;
    walls_config_x.line_width = width;
    walls_config_x.min_val = min_value;
    walls_config_x.max_val = max_value;

    Draw_Walls_Params walls_config_y = walls_config_x;
    walls_config_y.is_x_dir = false;

    isize count = 0;
    sim_make_face_vertices(g_cuda_vertices, &count, g_cuda_vertices_count, faces_x_span, walls_config_x);
    draw_vertices(g_cuda_vertices, count, NULL, NULL);
    sim_make_face_vertices(g_cuda_vertices, &count, g_cuda_vertices_count, faces_y_span, walls_config_y);
    draw_vertices(g_cuda_vertices, count, NULL, NULL);
}

void draw_walls(Sim_Face_Values* faces_x, Sim_Face_Values* faces_y, uint32_t intake_color, uint32_t outake_color, uint32_t wall_color, isize nx, isize ny, float width)
{
    if(g_cuda_vertices == NULL) 
        CUDA_TEST(cudaMalloc(&g_cuda_vertices, (size_t) g_cuda_vertices_count*6*sizeof(Sim_Color_Vertex)));

    Strided_2D_Span faces_x_span = {0};
    faces_x_span.data = (uint8_t*) faces_x + offsetof(Sim_Face_Values, average.ro);;
    faces_x_span.nx = nx + 1;
    faces_x_span.ny = ny;
    faces_x_span.stride = sizeof *faces_x;
    faces_x_span.pitch = faces_x_span.nx*(isize) sizeof *faces_x;

    Strided_2D_Span faces_y_span = {0};
    faces_y_span.data = (uint8_t*) faces_y + offsetof(Sim_Face_Values, average.ro);
    faces_y_span.nx = nx;
    faces_y_span.ny = ny + 1;
    faces_y_span.stride = sizeof *faces_y;
    faces_y_span.pitch = faces_y_span.nx*(isize) sizeof *faces_y;

    Strided_2D_Span flags_x_span = faces_x_span;
    flags_x_span.data = (uint8_t*) faces_x + offsetof(Sim_Face_Values, flags);

    Strided_2D_Span flags_y_span = faces_y_span;
    flags_y_span.data = (uint8_t*) faces_y + offsetof(Sim_Face_Values, flags);

    Sim_Flags intake_flags = SIM_SET_DER_RO | SIM_SET_VAL_UX | SIM_SET_VAL_UY;
    Sim_Flags outake_flags = SIM_SET_VAL_RO | SIM_SET_DER_UX | SIM_SET_DER_UY;

    Draw_Walls_Params walls_config_base = {0};
    walls_config_base.is_x_dir = true;
    walls_config_base.dx = 1.0f/nx;
    walls_config_base.dy = 1.0f/ny;
    walls_config_base.line_width = width;
    walls_config_base.min_val = 0;
    walls_config_base.max_val = 1;
    walls_config_base.use_static_color = true;

    /*
    marking of outside cells
    adding decorative flags
    drawing of wall types
    zooming in and out/panning
    
    */

    Draw_Walls_Params walls_config_x_intake = walls_config_base;
    walls_config_x_intake.is_x_dir = true;
    walls_config_x_intake.static_color = intake_color;

    Draw_Walls_Params walls_config_x_outake = walls_config_base;
    walls_config_x_outake.is_x_dir = true;
    walls_config_x_outake.static_color = outake_color;
    
    Draw_Walls_Params walls_config_y_intake = walls_config_base;
    walls_config_y_intake.is_x_dir = false;
    walls_config_y_intake.static_color = intake_color;

    Draw_Walls_Params walls_config_y_outake = walls_config_base;
    walls_config_y_outake.is_x_dir = false;
    walls_config_y_outake.static_color = outake_color;

    isize count = 0;
    sim_make_face_vertices_flagged(g_cuda_vertices, &count, g_cuda_vertices_count, faces_x_span, flags_x_span, intake_flags, walls_config_x_intake);
    sim_make_face_vertices_flagged(g_cuda_vertices, &count, g_cuda_vertices_count, faces_x_span, flags_x_span, outake_flags, walls_config_x_outake);
    sim_make_face_vertices_flagged(g_cuda_vertices, &count, g_cuda_vertices_count, faces_y_span, flags_y_span, intake_flags, walls_config_y_intake);
    sim_make_face_vertices_flagged(g_cuda_vertices, &count, g_cuda_vertices_count, faces_y_span, flags_y_span, outake_flags, walls_config_y_outake);
    draw_vertices(g_cuda_vertices, count, NULL, NULL);
}

void draw_colormap(int width, int height, float min, float max, bool linear_filtering, const Real* cuda_memory, const Sim_Flags* flags_or_null)
{
    enum { MAX_TEXTURES = 32 };
    enum { TYPE_VALUE, TYPE_FLAG };
    typedef struct {
        int width;
        int height;
        int type;
        GLuint handle;
    } Texture;

    static Texture g_textures[MAX_TEXTURES] = {0}; 
    static int    g_used_texture_count = 0;
    static void*  g_cpu_memory = NULL;
    static size_t g_cpu_memory_size = 0;

    int val_tex_index = -1;
    int flag_tex_index = -1;
    {
        for(int i = 0; i < g_used_texture_count; i++)
            if(g_textures[i].width == width && g_textures[i].height == height && g_textures[i].type == TYPE_VALUE) {
                val_tex_index = i;
                break;
            }

        if(val_tex_index == -1)
        {
            if(g_used_texture_count >= MAX_TEXTURES)
            {
                LOG_ERROR("opengl", "too many curently managed cuda resources!");
                return;
            }

            //Create a new texture and register it as cuda resource
            Texture texture = {width, height, TYPE_VALUE};
            glGenTextures(1, &texture.handle);
            glBindTexture(GL_TEXTURE_2D, texture.handle);

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_R32F, // internal format
                width, 
                height, 
                0, 
                GL_RED, // acess format
                GL_FLOAT, //data type
                NULL);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear_filtering ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear_filtering ? GL_LINEAR : GL_NEAREST);
            glGenerateMipmap(GL_TEXTURE_2D);

            val_tex_index = g_used_texture_count++;
            g_textures[val_tex_index] = texture;
        }
    }

    if(flags_or_null) {
        for(int i = 0; i < g_used_texture_count; i++)
            if(g_textures[i].width == width && g_textures[i].height == height && g_textures[i].type == TYPE_FLAG) {
                flag_tex_index = i;
                break;
            }

        if(flag_tex_index == -1)
        {
            if(g_used_texture_count >= MAX_TEXTURES)
            {
                LOG_ERROR("opengl", "too many curently managed cuda resources!");
                return;
            }

            //Create a new texture and register it as cuda resource
            Texture texture = {width, height, TYPE_FLAG};
            glGenTextures(1, &texture.handle);
            glBindTexture(GL_TEXTURE_2D, texture.handle);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_R16UI, // internal format
                width, 
                height, 
                0, 
                GL_RED, // acess format
                GL_UNSIGNED_SHORT, //data type
                NULL);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            flag_tex_index = g_used_texture_count++;
            g_textures[flag_tex_index] = texture;
        }
    }

    size_t pixel_count = (size_t) width * (size_t) height;
    size_t required_bytes = MAX(pixel_count * sizeof(float), pixel_count * sizeof(Sim_Flags));
    if(g_cpu_memory_size < required_bytes) {
        g_cpu_memory_size = required_bytes;
        g_cpu_memory = realloc(g_cpu_memory, g_cpu_memory_size);
    }

    unsigned val_texture_handle = (unsigned) -1;
    unsigned flag_texture_handle = (unsigned) -1;
    Sim_Flags flagged_mask = 0;
    {
        Texture val_texture = g_textures[val_tex_index];
        sim_modify_float((Real*) cuda_memory, (float*) g_cpu_memory, pixel_count, MODIFY_DOWNLOAD);
        glBindTexture(GL_TEXTURE_2D, val_texture.handle);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, g_cpu_memory);

        val_texture_handle = (unsigned) val_texture.handle;
    }

    if(flags_or_null) {
        Texture flag_texture = g_textures[flag_tex_index];
        cudaMemcpy(g_cpu_memory, flags_or_null, pixel_count*sizeof(Sim_Flags), cudaMemcpyDeviceToHost);

        // sim_modify_float((Real*) cuda_memory, (float*) g_cpu_memory, pixel_count, MODIFY_DOWNLOAD);
        glBindTexture(GL_TEXTURE_2D, flag_texture.handle);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, g_cpu_memory);
        
        flagged_mask = (Sim_Flags) -1;
        flag_texture_handle = (unsigned) flag_texture.handle;
    }

    uint32_t flagged_color = 0xFFFFFF;
    draw_gl_colormap(val_texture_handle, flag_texture_handle, flagged_mask, flagged_color, min, max, NULL, NULL);

    glFinish();
}

void gl_init(void* load_function)
{
    int version = gladLoadGL((GLADloadfunc) load_function);
    TEST(version != 0, "Failed to load opengl with glad");
    LOG_INFO("opengl", "initialized opengl");
    
    gl_debug_output_enable();
}

#endif