import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100


cells_ro = np.zeros(N)
cells_ux = np.zeros(N+1)

cells_new_ro = np.zeros(N)
cells_new_ux = np.zeros(N)

faces_avg_ro = np.zeros(N+1)
faces_avg_ux = np.zeros(N+1)

faces_upw_ro = np.zeros(N+1)
faces_upw_ux = np.zeros(N+1)

faces_der_ro = np.zeros(N+1)
faces_der_ux = np.zeros(N+1)
faces_r = np.zeros(N+1)

# BOUNDARY CONDITIONS

#inflow: set val ux, set der ro ??? set val ro
#outflow: set der ux, set val ro

boundary_condition_inflow_ro = "der"
boundary_condition_inflow_ux = "val"

boundary_condition_outlow_ro = "val"
boundary_condition_outflow_ux = "der"

R_spec = 287
T = 272 + 20
c_sound = 343
piso_iters = 2

inflow_ux = 1
inflow_ro = 1

outflow_ro = 1
outflow_ux = 0

# Physics
width = 0.5
dx = width/N
dt = 2e-7

ini_ro = 1
ini_ux = 0
ini_ux_last = None

t0 = 0
t1 = 20

lam = -1.5308e-2
mu = 1.3059e-2

show_interval = 0.00
multistep = 100

def set_initial_conds():
    for i in range(0, N):
        cells_ro[i] = ini_ro
        cells_ux[i] = ini_ux

    if ini_ux_last != None:
        cells_ux[-1] = ini_ux_last

def get_cfl():
    return np.max(np.abs(cells_ux)*dt/dx)

def get_mach_number():
    return np.max(np.abs(cells_ux)/c_sound)

# p = ρ*Rspec*T
# => ρ = p / (Rspec*T)
def ro_from_pressure(p):
    return p / (R_spec*T)

# These functions return "stencils" of coefficients
# which can later be used to evaluate the given equation
# explicitly or form a matrix equation and evaluate implciitly.
# The format is little strange but makes sense.
# 
# Stencil is array of 4 item:
# stencil[0] - west neighbour coeficient
# stencil[1] - evaluated cell coeficient
# stencil[2] - east neighbour coeficient
# stencil[3] - source term which is shifted to right hand side
# stencil[4] - value (see below)
# 
# so together one can convert the stencil to value by doing
# value = var_west*stencil[0] + var_curr*stencil[1] + var_east*stencil[2] + stencil[3]
# 
# The advantage of this format is that one can calculate with stencils the same way one 
# would calculate with the produced values (as long as only linear transfomations are used).

FACE_AREA = dx
CELL_AREA = dx*dx
rho = 1

def H(u, i, eval=False):
    upw_stencil_w = np.zeros(4)
    upw_stencil_e = np.zeros(4)
    der_stencil_w = np.zeros(4)
    der_stencil_e = np.zeros(4)

    u_w = 0
    u_c = 0
    u_e = 0

    assert i >= 0 and i <= N
    if i == 0:
        u_c = u[i]
        u_e = u[i+1]

        # upwind u
        upw_stencil_w[3] = inflow_ux

        if u_e - u_c >= 0:
            upw_stencil_e[1] += u_c
        else:
            upw_stencil_e[2] += u_e

        # diffusion
        der_stencil_w[3] = -2/dx*inflow_ux
        der_stencil_w[1] = 2/dx

        der_stencil_e[1] = -1/dx
        der_stencil_e[2] = 1/dx
    elif i == N:
        u_w = u[i-1]
        u_c = u[i]

        # upwind u
        if u_c - u_w >= 0:
            upw_stencil_w[0] += u_w
        else:
            upw_stencil_w[1] += u_c

        upw_stencil_e[1] += u_c

        # diffusion
        der_stencil_w[0] = -1/dx
        der_stencil_w[1] = 1/dx

        der_stencil_e[1] = 0
        der_stencil_e[2] = 0
    else:
        u_w = u[i-1]
        u_c = u[i]
        u_e = u[i+1]

        # upwind u
        if u_c - u_w >= 0:
            upw_stencil_w[0] += u_w
        else:
            upw_stencil_w[1] += u_c

        if u_e - u_c >= 0:
            upw_stencil_e[1] += u_c
        else:
            upw_stencil_e[2] += u_e

        # diffusion
        der_stencil_w[0] = -1/dx
        der_stencil_w[1] = 1/dx

        der_stencil_e[1] = -1/dx
        der_stencil_e[2] = 1/dx

    H = -rho*(upw_stencil_e - upw_stencil_w)*FACE_AREA \
        +lam*(der_stencil_e - der_stencil_e)*FACE_AREA
    
    if i == 0:
        assert H[0] == 0
    if i == N:
        assert H[2] == 0

    if eval:
        return u_w*H[0] + u_c*H[1] + u_e*H[2] + H[3]
    else:
        return H


def delta_u(u):
    delta = np.zeros_like(u)
    delta[1:-1] = (u[2:] - u[:-2])*(FACE_AREA/2)
    delta[0] = (u[1] + u[0])/2*FACE_AREA - inflow_ux*FACE_AREA
    delta[-1] = 0
    return None

def delta_H(H, u):
    delta = np.zeros_like(H)
    delta[1:-1] = (H[2:] - H[:-2])*(FACE_AREA/2)
    delta[0] = (H[1] + H[0])/2*FACE_AREA - inflow_ux*FACE_AREA
    delta[-1] = 0
    return None

def step(u, p):

    # precalculate common coeffs
    S = 0
    b = rho*CELL_AREA/dt*u + S*CELL_AREA
    dx_b = delta_u(b)

    # predictor
    for i in range(0, N+1):
        h = H(u, i)

        A_w = -h[0]
        A_c = rho/dt - h[1]
        A_e = -h[2]
        b = u[i] - (p[i] - p[i-1])*FACE_AREA + h[4]

    # solve somehow...
    u_star = np.zeros_like(u)

    corrector_count = 2
    
    for k in range(corrector_count):
        #pressure poisson equation matrix
        for i in range(0, N+1):
            b = H(u_star, i+1, eval=True) - H(u_star, i, eval=True) \
                +  rho/dt*(u[i+1] - u[i])
            A_w = 1*dx
            A_c = -2*dx
            A_e = 1*dx

        # solve using CG
        p_star_next = np.zeros_like(p)

        u_star_next = np.zeros_like(u)
        for i in range(0, N):
            h = H(u_star)
            u_star_next[i] = 


x_ro = ro_from_pressure(1e5)

def step():
    # caclulate faces
    assert N > 0

    inflow_mass = 0
    outflow_mass = 0
    for i in range(0, N+1):
        #        face[i]
        # cell[i-1] | cellp[i]
        celln_ro = cells_ro[i-1] if i > 0 else 0
        celln_ux = cells_ux[i-1] if i > 0 else 0

        cellp_ro = cells_ro[i] if i < N else 0
        cellp_ux = cells_ux[i] if i < N else 0

        avg_ro = 0
        avg_ux = 0

        upw_ro = 0
        upw_ux = 0

        der_ro = 0
        der_ux = 0

        if i == 0:
            if boundary_condition_inflow_ro == "val":
                avg_ro = inflow_ro
                upw_ro = inflow_ro
                der_ro = (cellp_ro - inflow_ro)/(dx/2) 
            else:
                avg_ro = cellp_ro
                upw_ro = cellp_ro
                der_ro = 0

            avg_ux = inflow_ux
            upw_ux = inflow_ux
            der_ux = (cellp_ux - inflow_ux)/(dx/2) 

            inflow_mass = avg_ro*avg_ux

        elif i == N:
            if boundary_condition_outlow_ro == "val":
                avg_ro = outflow_ro
                upw_ro = outflow_ro 
                # upw_ro = celln_ro
                der_ro = (outflow_ro - celln_ro)/(dx/2) 
            else:
                avg_ro = celln_ro
                upw_ro = celln_ro
                der_ro = 0

            avg_ux = celln_ux
            upw_ux = celln_ux
            der_ux = 0

            outflow_mass = avg_ro*avg_ux

            # avg_ux *= inflow_mass/outflow_mass
            # upw_ux *= inflow_mass/outflow_mass

        # interior cell
        else:
            avg_ro = (cellp_ro + celln_ro)/2
            avg_ux = (cellp_ux + celln_ux)/2

            upw_ro = celln_ro
            upw_ux = celln_ux
            
            der_ro = (cellp_ro - celln_ro)/dx
            der_ux = (cellp_ux - celln_ux)/dx


        faces_avg_ro[i] = avg_ro
        faces_avg_ux[i] = avg_ux
        faces_upw_ro[i] = upw_ro
        faces_upw_ux[i] = upw_ux
        faces_der_ro[i] = der_ro
        faces_der_ux[i] = der_ux

    # calculate cells
    for i in range(0, N):
        # face[i] | face[i+1]
        #   | cell[i] |

        fw_avg_ro = faces_avg_ro[i]
        fw_avg_ux = faces_avg_ux[i]
        fw_upw_ro = faces_upw_ro[i]
        fw_upw_ux = faces_upw_ux[i]
        fw_der_ro = faces_der_ro[i]
        fw_der_ux = faces_der_ux[i]

        fe_avg_ro = faces_avg_ro[i+1]
        fe_avg_ux = faces_avg_ux[i+1]
        fe_upw_ro = faces_upw_ro[i+1]
        fe_upw_ux = faces_upw_ux[i+1]
        fe_der_ro = faces_der_ro[i+1]
        fe_der_ux = faces_der_ux[i+1]

        ro = cells_ro[i]
        ux = cells_ux[i]

        ax = 0
            
        dt_ro = -(fe_avg_ro*fe_avg_ux - fw_avg_ro*fw_avg_ux)/dx 
        
        ro_dt_ux = \
            -(fe_upw_ro*fe_upw_ux*fe_avg_ux - fw_upw_ro*fw_upw_ux*fw_avg_ux)/dx \
            +(lam + 2*mu)*(fe_der_ux - fw_der_ux)/dx \
            -ux*dt_ro + ro*ax \
            -R_spec*T*(fe_avg_ro - fw_avg_ro)/dx 

        dt_ux = ro_dt_ux/ro

        new_ro = ro + dt_ro*dt
        new_ux = ux + dt_ux*dt

        cells_new_ro[i] = new_ro
        cells_new_ux[i] = new_ux

def graph():
    global cells_ro
    global cells_new_ro
    global cells_ux
    global cells_new_ux

    # Setup
    cell_centers = (np.arange(N) + 0.5)*(width/N)
    face_positions = np.arange(N+1)*(width/N)

    set_initial_conds()

    plt.ion()  # Turn on interactive mode

    fig, (ax_ux, ax_ro) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ux_cells, = ax_ux.plot(cell_centers, cells_ux, label='ux cells')
    # ux_faces, = ax_ux.plot(face_positions, faces_avg_ux, 's', label='ux faces')

    ax_ux.set_ylim(0, 2)
    ax_ux.set_xlabel('x')
    ax_ux.set_ylabel('ux')
    ax_ux.legend()
    ax_ux.grid(True)

    ro_cells, = ax_ro.plot(cell_centers, cells_ro, label='ro cells')
    # ro_faces, = ax_ro.plot(face_positions, faces_avg_ro, 's', label='ro faces')
    ax_ro.set_ylim(0.99, 1.01)
    ax_ro.set_xlabel('x')
    ax_ro.set_ylabel('ro')
    ax_ro.legend()
    ax_ro.grid(True)

    t = t0
    iter = 0
    while t < t1:
        step()

        cells_ro, cells_new_ro = cells_new_ro, cells_ro
        cells_ux, cells_new_ux = cells_new_ux, cells_ux

        if iter % multistep == 0:
            ux_cells.set_ydata(cells_ux)
            # ux_faces.set_ydata(faces_avg_ux)

            ro_cells.set_ydata(cells_ro)
            # ro_faces.set_ydata(faces_avg_ro)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            cfl = get_cfl()
            Ma = get_mach_number()
            ax_ux.set_title(f"t = {float(t):.6} CFL = {cfl:.2} Ma = {Ma:.2}")
            time.sleep(show_interval) 

        t += dt
        iter += 1

    plt.ioff()  # Turn off interactive mode after loop ends
    plt.tight_layout()
    plt.show()

graph()