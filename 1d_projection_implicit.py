import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

N = 100

R_spec = 287
T = 272 + 20
c_sound = 343

width = 1
dx = width/N
dt = 2e-1

rho = 1
ini_p = 1
ini_ux = 0

t0 = 0
t1 = 20

lam = -1.5308e-2
mu = 1.3059e-2

show_interval = 0.00
multistep = 1

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

class Boundary:
    type = "val"
    value = 0.0

    def __init__(self, type, value=0):
        self.type = type
        self.value = value

def dirichlet(val=0):
    return Boundary("val", val)

def neumann(val=0):
    return Boundary("der", val)

def apply_stencil(stencil, field):
    assert len(stencil) == 4
    assert len(stencil[0]) == len(field)

    out = np.zeros_like(field)
    out[1:] += stencil[0, :-1]*field[:-1]
    out += stencil[1]*field 
    out[:-1] += stencil[2, 1:]*field[1:]
    out += stencil[3] 
    return out

def set_stencil(A, row, w, c, e, b):
    rows = len(A[0])
    if row < 0:
        row = rows + row

    if row in range(1, rows):
        A[0, row-1] = w
    if row in range(0, rows):
        A[1, row+0] = c
        A[3, row+0] = b
    if row in range(0, rows-1):
        A[2, row+1] = e


def central(BC, phi=None):
    A = np.zeros((4, N))
    A[0, :-2] += -1/(2*dx)
    A[2, 2:] += 1/(2*dx)

    V = BC[0].value
    if BC[0].type == "val":
        set_stencil(A, 0, 0, 1/(2*dx), 1/(2*dx), -V/dx)
    elif BC[0].type == "der":
        set_stencil(A, 0, 0, -1/(2*dx), 1/(2*dx), V/2)

    V = BC[1].value
    if BC[1].type == "val":
        set_stencil(A, -1, -1/(2*dx), -1/(2*dx), 0, V/dx)
    elif BC[1].type == "der":
        set_stencil(A, -1, -1/(2*dx), 1/(2*dx), 0, V/2)

    if phi is not None:
        return apply_stencil(A, phi)
    else:
        return A
    
def central_second(BC, phi=None):
    A = np.zeros((4, N))
    A[0, :-2] += 1/dx**2
    A[1, 1:-1] += -2/dx**2
    A[2, 2:] += 1/dx**2

    V = BC[0].value
    if BC[0].type == "val":
        set_stencil(A, 0, 0, -3/dx**2, 1/dx**2, 2*V/dx**2)
    elif BC[0].type == "der":
        set_stencil(A, 0, 0, -1/dx**2, 1/dx**2, -V/dx)

    V = BC[1].value
    if BC[1].type == "val":
        set_stencil(A, -1, 1/dx**2, -3/dx**2, 0, 2*V/dx**2)
    elif BC[1].type == "der":
        set_stencil(A, -1, 1/dx**2, -1/dx**2, 0, V/dx)

    if phi is not None:
        return apply_stencil(A, phi)
    else:
        return A

def upwind(u, BC_u, BC_phi, phi=None):
    A = np.zeros((4, len(u)))
    upw_dir = u[1:-1] >= 0

    #                       C                   W                           E               C 
    # delta[1:-1] = upw_dir*(u[1:-1]*v[1:-1] - u[:-2]*v[:-2])/dx + (upw_dir - 1)*(u[2:]*v[2:] - u[1:-1]*v[1:-1])/dx
    A[0, :-2] += upw_dir*(-u[:-2]/dx)
    A[1, 1:-1] += upw_dir*u[1:-1]/dx - (upw_dir - 1)*u[1:-1]/dx  
    A[2, 2:] += (upw_dir - 1)*u[2:]/dx

    # reconstruct outside cell uw based on u BC 
    uw = 0
    uc = u[0]
    ue = u[1]
    V = BC_u[0].value
    if BC_u[0].type == "val":
        uw = 2*V - uc
    elif BC_u[0].type == "der":
        uw = uc - dx*V
    
    # apply phi BC (if necessary)
    if uc >= 0:
        if BC_phi[0].type == "val":
            set_stencil(A, 0, 0, (uc + uw)/(dx), 0, -2*V*uw/dx)
        elif BC_phi[0].type == "der":
            set_stencil(A, 0, 0, (uc - uw)/(dx), 0, V*uw)
    else:
        set_stencil(A, 0, 0, -1/dx, 1/dx, 0)

    # reconstruct outside cell uw based on u BC 
    uw = u[-2]
    uc = u[-1]
    ue = 0
    V = BC_u[1].value
    if BC_u[1].type == "val":
        ue = 2*V - uc
    elif BC_u[1].type == "der":
        ue = dx*V + uc

    # apply phi BC (if necessary)
    if uc < 0:
        if BC_phi[-1].type == "val":
            set_stencil(A, -1, 0, -(ue + uc)/(dx), 0, 2*V*ue/dx)
        elif BC_phi[-1].type == "der":
            set_stencil(A, -1, 0, (ue - uc)/(dx), 0, V*ue)
    else:
        set_stencil(A, -1, -1/dx, 1/dx, 0, 0)

    if phi is not None:
        return apply_stencil(A, phi)
    else:
        return A
    
def H(u, BC_u, BC_phi, phi=None):
    A = upwind(u, BC_u=BC_u, BC_phi=BC_phi, phi=phi) #advection
    D = central_second(BC_phi, phi=phi) #diffusion
    return -rho*A + mu*D

def stencil_to_sparse_matrix(A):
    return sp.sparse.diags([A[0, :-1], A[1], A[2, 1:]], [-1, 0, 1])

def step(u, p, BC_u, BC_p, method="implicit"):
    S = 0

    u_star = None
    u_strange = None

        # assert info == 0
    # Classic projection method (explicit advection diffusion)
    if method == "explicit":
        u_star = u + dt/rho*( \
            H(u, BC_u, BC_u, phi=u) \
            + S \
        ) \
        
    # Implicit projection method (implicit advection diffusion)
    elif method == "implicit":
        h = H(u, BC_u, BC_u)

        U = np.zeros((3, len(u)))
        U[1] += rho/dt
        U -= h[:3]

        b = rho/dt*u + S + h[3]

        u_star, info = sp.sparse.linalg.cgs(stencil_to_sparse_matrix(U), b)
        if info != 0:
            u_strange = upwind(u, BC_u=BC_u, BC_phi=BC_u, phi=u) #advection


    # "Stable fluids" method (explicit advection, implicit diffusion)
    elif method == "stable":
        A = upwind(u, BC_u=BC_u, BC_phi=BC_u, phi=u) 
        u_adv = u + dt*A

        D = mu*central_second(BC_u) #diffusion
        U = -D
        U[1] += rho/dt

        b = rho/dt*u + S + D[3]

        u_star, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(U), b)

    else:
        assert(False)

    P = central_second(BC_p)
    b = rho/dt*central(BC_u, phi=u_star) + P[3]

    p_next = np.zeros_like(p)
    p_next, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(P), b)
    assert info == 0

    # u_next = u_star
    u_next = u_star - dt/rho*central(BC_p, p_next)

    return (u_next, p_next)

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

    for i in range(len(u)//4, len(u)//2):
        u[i] = 1

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
        # method = "explicit"
        method = "implicit"
        # method = "stable"
        BC_u = (dirichlet(inflow_ux), neumann(0))
        BC_p = (neumann(0),           dirichlet(0))

        u_next, p_next = step(u, p, BC_u, BC_p, method=method)

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

            div_u = np.linalg.norm(central(BC_u, u))

            # ax_ux.set_title(f"t = {float(t):.6} CFL = {cfl:.2} Ma = {Ma:.2}")
            ax_ux.set_title(f"t = {float(t):.6} div(u) = {div_u:.4e}")
            time.sleep(show_interval) 



        t += dt
        iter += 1

    plt.ioff()  # Turn off interactive mode after loop ends
    plt.show()

graph()