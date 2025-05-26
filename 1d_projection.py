import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100

boundary_condition_inflow_p = "der"
boundary_condition_inflow_ux = "val"

boundary_condition_outlow_p = "val"
boundary_condition_outflow_ux = "der"

a = np.array([np.zeros(10), np.ones(10), np.zeros(10)])
a += 1

R_spec = 287
T = 272 + 20
c_sound = 343

inflow_ux = 1
inflow_p = 1

outflow_p = 0
outflow_ux = 0

width = 1
dx = width/N
dt = 2e-3

rho = 1
ini_p = 1
ini_ux = 0

t0 = 0
t1 = 2

lam = -1.5308e-5
mu = 1.3059e-5

show_interval = 0.00
multistep = 100

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

def upwind(u):
    mask = u[1:-1] >= 0
    delta = np.zeros_like(u)
    delta[1:-1] = mask*(u[1:-1]**2 - u[:-2]**2)/dx + (mask - 1)*(u[2:]**2 - u[1:-1]**2)/dx
    # delta[1:-1] = (u[1:-1]**2 - u[:-2]**2)/dx

    if u[0] >= 0:
        delta[0] = (u[0] - inflow_ux)/(dx/2)
    else:
        delta[0] = (u[1] - u[0])/dx

    delta[-1] = 0
    return delta

def der_u(u):
    delta = np.zeros_like(u)
    delta[1:-1] = (u[2:] - u[:-2])/(2*dx)
    delta[0] = ((u[1] + u[0])/2 - inflow_ux) / dx
    delta[-1] = 0 
    return delta

def der_p(p):
    delta = np.zeros_like(p)
    delta[1:-1] = (p[2:] - p[:-2])/(2*dx)
    delta[0] = 0
    delta[-1] = (outflow_p - (p[-2] + p[-2])/2) / dx
    return delta

def der_der_u(u):
    delta = np.zeros_like(u)
    delta[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/(dx*dx)
    delta[0] = ((u[1] - u[0])/dx - (u[0] - inflow_ux)/(dx/2)) / dx
    delta[-1] = (0 - (u[-1] - u[-2])/dx) / dx
    return delta

main_diag = -2/dx**2 * np.ones(N)
off_diag = 1/dx**2 * np.ones(N - 1)
A = sp.sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1])

def step(u):
    S = 0
    u_star = u + dt/rho*( \
        -rho*upwind(u) \
        +mu*der_der_u(u) \
        + S \
    ) \
    
    b = rho/dt*der_u(u_star)
    p, info = sp.sparse.linalg.cg(A, b)
    assert info == 0

    u_next = u_star - dt/rho*der_p(p)

    return (u_next, p)

def graph():
    global inflow_ux

    # Setup
    cell_centers = (np.arange(N) + 0.5)*(width/N)
    face_positions = np.arange(N+1)*(width/N)

    plt.ion()  # Turn on interactive mode
    # plt.tight_layout()

    fig, (ax_ux, ax_p) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    p = np.zeros(N) + ini_p
    u = np.zeros(N) + ini_ux

    # for i in range(len(u)//4, len(u)//2):
        # u[i] = 1

    u_cells, = ax_ux.plot(cell_centers, u, label='u')
    # u_faces, = ax_ux.plot(face_positions, faces_avg_ux, 's', label='ux faces')

    ax_ux.set_ylim(0, 2)
    ax_ux.set_xlabel('x')
    ax_ux.set_ylabel('ux')
    ax_ux.legend()
    ax_ux.grid(True)

    p_cells, = ax_p.plot(cell_centers, p, label='p')
    # p_faces, = ax_p.plot(face_positions, faces_avg_p, 's', label='ro faces')

    ax_p.set_ylim(0, 2)
    ax_p.set_xlabel('x')
    ax_p.set_ylabel('p')
    ax_p.legend()
    ax_p.grid(True)

    t = t0
    iter = 0
    while t < t1:
        inflow_ux = min(1, t)
        # inflow_ux = 0

        u_next, p_next = step(u)

        u, u_next = u_next, u
        p, p_next = p_next, p
        if iter % multistep == 0:
            u_cells.set_ydata(u)
            p_cells.set_ydata(p)
            
            max_p = np.max(p)
            min_p = np.min(p)
            fig.canvas.draw()
            fig.canvas.flush_events()
            cfl = get_cfl(u)
            Ma = get_mach_number(u)
            div = np.linalg.norm(der_u(u))

            # ax_ux.set_title(f"t = {float(t):.6} CFL = {cfl:.2} Ma = {Ma:.2}")
            ax_ux.set_title(f"t = {float(t):.6} div(u) = {div:.4e}")
            time.sleep(show_interval) 



        t += dt
        iter += 1

    plt.ioff()  # Turn off interactive mode after loop ends
    plt.show()

graph()