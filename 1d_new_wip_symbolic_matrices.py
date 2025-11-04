
# Solve div x = B assuming Neumann BC with:
#   partial x = N
# 
#----------------------------------------
#  G1  |   W   :   C   :    E   |   G2  |
#------^------------------------^--------
#      |                        |
#   boundary                 boundary
# 
#  G1, G2 are ficticious "ghost" cells
#  with value derived such that the boundary condition is satisfied
#  by the desired differential operator. Because of 0 Nuemann we want:
#       (W - G1)/h = N  
#       (G2 - E)/h = N
#  from which we derive that G1 = W - Nh, G2 = E + Nh.
#   
#  Writing equations for (non-ghost) cells:
#  W:  (C - G1)/2h = (W - C)/2h + N/2 =!= B1
#  C:  (E - W )/2h = (E - W)/2h       =!= B2
#  E:  (G2 - C)/2h = (E - C)/2h + N/2 =!= B3
# 
#             | Write as a matrix equation
#             V 
# 
#       | 1   -1    | |W|   |B1|   |N/2|
#  1/2h*|-1        1|*|C| = |B2| - |   |
#       |     -1   1| |E|   |B3|   |N/2|
#           
#             | Transform into a sparse matrix
#             V 
# 
#  "Stencil" (sparse matrix) [w, c, e, b] where 
#    - w is below diagonal 
#    - c is on diagonal 
#    - e is above diagonal 
#    - b is right hand term that appears due to boundary conditions 
#   and if no w/e logically exist then we set 0 (first/last row)
#  
#  a := 1/2h
#  Row1: [0,  a, -a, N/2]
#  Row2: [-a, 0,  a,   0]
#  Row3: [-a, a,  0, N/2]
# 
#  Solve for [W,C,E] using some sparse matrix solver Bi-CG.
#            
#            vs.
#  
#  First we "ghosts" field [W,C,E] to add extrapolated ghost cells
#  and then calculate using non-square matrices. These matrices have
#  however more regular structure.
#  If the expansion is linear (here it is) then it can be expressed as 
#  some matrix operation acting on the ghostsed field. If we multiply
#  the expansion matrix and the ghostsed equation matrix we will get exactly
#  the same matrix as in the classic approach.
# 
#                           |G1|   
#       |-1       1       | |W|   |B1|
#  1/2h*|    -1       1   |*|C| = |B2|
#       |         -1     1| |E|   |B3|
#                           |G2|   
# 
#  Why not do the expansion approach when it looks a lot simpler? 
#  On this particular approach there is not much of a reason not to, 
#  however consider wanting to operate on the matrix in some fashion. 
#  For example dividing by the 
#            

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass

N = 100

R_spec = 287
T = 272 + 20
c_sound = 343

width = 1
dx = width/N
dt = 2e-2

rho = 1
ini_p = 0
ini_ux = 0

t0 = 0
t1 = 20

lam = -1.5308e-3
mu = 1.3059e-3

show_interval = 0.00
multistep = 1

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

def expand(phi:np.ndarray, w:float = 0.0, e:float = 0.0) -> np.ndarray:
    out = np.empty(phi.size + 2, dtype=phi.dtype)
    out[-1:1] = phi
    out[0] = w
    out[-1] = e

def contract(phi:np.ndarray) -> np.array:
    return phi[-1,1]

def dirichlet(val=0.0): return Boundary("val", val)
def neumann(val=0.0):   return Boundary("der", val)

def apply(stencil:np.ndarray, field:np.ndarray, parts:list | None = None) -> np.ndarray:
    out = np.empty_like(field)
    if parts is None:
        out = sp.ndimage.convolve1d(field, stencil[:3], mode='constant', cval=0.0)
        out[1:-1] += stencil[3]
        return out
    else:
        out = np.zeros_like(field)
        if 0 in parts: out[1:-1] += stencil[0]*field[:-2]
        if 1 in parts: out[1:-1] += stencil[1]*field[1:-1]
        if 2 in parts: out[1:-1] += stencil[2]*field[2:]
        if 3 in parts: out[1:-1] += stencil[3]
        return out

def set_stencil(A, row, w, c, e, b):
    A[0, row] = w
    A[1, row] = c
    A[2, row] = e
    A[3, row] = b

@dataclass
class Boundary:
    type: str
    value: float

Boundaries = Tuple[Boundary, Boundary]
class Grid:
    N: int
    dx: float

    central_stencil: np.ndarray
    forward_stencil: np.ndarray
    backward_stencil: np.ndarray
    central2_stencil: np.ndarray
    central2_kernel: np.ndarray
    def __init__(self, N:int, dx:float):
        self.N = N
        self.dx = dx

        self.forward_stencil = np.zeros((4, N))
        self.forward_stencil[1] += -1/dx
        self.forward_stencil[2] += 1/dx

        self.backward_stencil = np.zeros((4, N))
        self.backward_stencil[0] += -1/dx
        self.backward_stencil[1] += 1/dx

        self.central_stencil = np.zeros((4, N))
        self.central_stencil[0] += -1/(2*dx)
        self.central_stencil[2] += 1/(2*dx)
        
        self.central2_stencil = np.zeros((4, N))
        self.central2_stencil[0] += 1/dx**2
        self.central2_stencil[1] += -2/dx**2
        self.central2_stencil[2] += 1/dx**2

        self.central2_kernel = np.array([1, -2, 1]) / dx**2

    def central(self, phi:np.ndarray|None = None) -> np.ndarray:
        if phi is None:
            return np.copy(self.central_stencil)
        else:
            return np.gradient(phi, self.dx)
            
    def central2(self, phi:np.ndarray|None = None) -> np.ndarray:
        if phi is None:
            return np.copy(self.central2_stencil)
        else:
            return sp.ndimage.convolve1d(phi, self.central2_kernel, mode='constant', cval=0.0)

    def upwind(self, u:np.ndarray, phi:np.ndarray|None = None) -> np.ndarray:
        upw_dir = u[1:-1] >= 0
        if phi is None:
            return np.logical_not(upw_dir)*self.forward_stencil + upw_dir*self.backward_stencil
        else:
            difs = np.diff(phi, prepend=0.0, append=0.0)
            forward = difs[1:]
            backward = difs[:-1]
            return np.logical_not(upw_dir)[:, None]*forward + upw_dir[:, None]*backward
            
    def H(self, u:np.ndarray, phi:np.ndarray|None = None) -> np.ndarray:
        Adv = self.upwind(u, phi)
        Dif = self.central2(phi)
        return -rho*Adv + mu*Dif

def BC_ghost_west(g:Grid, BC:Boundary, val:float) -> float:
    if BC.type == "val":
        return 2*BC.value - val
    elif BC.type == "der":
        return -BC.value*g.dx + val
    return 0.0

def BC_ghost_east(g:Grid, BC:Boundary, val:float) -> float:
    if BC.type == "val":
        return 2*BC.value - val
    elif BC.type == "der":
        return BC.value*g.dx + val
    return 0.0

def BC_stencil_west(g:Grid, BC:Boundary, row:np.ndarray) -> np.ndarray:
    if BC.type == "val":
        transform = np.array([
            [0, -1, 0, 2*BC.value],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return transform.T @ row
    elif BC.type == "der":
        transform = np.array([
            [0, 1, 0, -BC.value*g.dx],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return transform.T @ row
    return row.copy()
    
def BC_stencil_east(g:Grid, BC:Boundary, row:np.ndarray) -> np.ndarray:
    if BC.type == "val":
        transform = np.array([
            [0, -1, 0, 2*BC.value],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return transform.T @ row
    elif BC.type == "der":
        transform = np.array([
            [0, 1, 0, BC.value*g.dx],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return transform.T @ row
    return row.copy()

def BC_ghosts(g:Grid, phi:np.ndarray, BCs:Boundaries):
    phi[0] = BC_ghost_west(g, BCs[0], phi[1])
    phi[-1] = BC_ghost_east(g, BCs[-1], phi[-2])

def BC_stencil(g:Grid, stencil:np.ndarray, BCs:Boundaries):
    stencil[:, 0] = BC_stencil_west(g, BCs[0], stencil[:, 0])
    stencil[:, -1] = BC_stencil_east(g, BCs[-1], stencil[:, -1])

def solve_stencil(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    sparse = sp.sparse.diags([A[0, :-1], A[1], A[2, 1:]], [-1, 0, 1])
    out, info = sp.sparse.linalg.cgs(sparse, b, maxiter=1000)
    if info != 0:
        cond_num = sparse_cond_num(sparse)
        print(f"{cond_num=}")
    return out

def piso_step(u:np.ndarray, p:np.ndarray, g:Grid, BCu:Boundaries , BCp:Boundaries, piso_iters=4, relax=0.4) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate common needed coeffs
    BC_ghosts(g, u, BCu)
    BC_ghosts(g, p, BCp)

    H = g.H(u)
    BC_stencil(g, H, BCu)

    S = 0
    A = H[1]
    B = H[3]
    alpha = rho/dt
    R = alpha*u + S + B

    # predictor
    A_pred = -H
    A_pred[1] += alpha 
    b_pred = -g.central(p) + R
    u_pred = solve_stencil(A_pred, b_pred)

    # corrector loop
    u_star = u_pred
    p_star = p
    for k in range(piso_iters):
        BC_ghosts(g, u_star, BCu)
        BC_ghosts(g, p_star, BCp)

        # Calc artificial field and its ghost cells to
        # match predictor/corrector eq
        L = apply(H, u_star, parts=[0,2]) + R
        L[ 0] = (alpha - A[ 0])*u[ 1] + (p_star[ 1] - p_star[ 0])/g.dx
        L[-1] = (alpha - A[-1])*u[-2] + (p_star[-1] - p_star[-2])/g.dx

        # posisson EQ for pressure
        A_poisson = g.central2()
        b_poisson = g.central(L)
        p_poisson = solve_stencil(A_poisson, b_poisson)

        # explicit update of velocity
        BC_ghosts(p_poisson, BCp)
        div_p_poisson = g.central(p_poisson)

        p_star_next = p_poisson
        u_star_next = (L - div_p_poisson)/(alpha - A)

        # use corrected 
        u_star = u_star*(1 - relax) + u_star_next*relax
        p_star = p_star*(1 - relax) + p_star_next*relax

    return (u_star, p_star)

def sparse_cond_num(A):
    try:
        norm_A = sp.sparse.linalg.norm(A)
        norm_A_inv = sp.sparse.linalg.norm(sp.sparse.linalg.inv(A))
        return norm_A*norm_A_inv
    except RuntimeError:
        return np.inf

def projection_step(u:np.ndarray, p:np.ndarray, g:Grid, BCu:Boundaries , BCp:Boundaries, method="implicit") -> Tuple[np.ndarray, np.ndarray]:
    S = 0
    u_star = None

    BC_ghosts(g, u, BCu)
    BC_ghosts(g, p, BCp)
    # Classic projection method (explicit advection diffusion)
    if method == "explicit":
        u_star = u + dt/rho*( 
            g.H(u, phi=u) 
            + S 
        ) 
    # Implicit projection method (implicit advection diffusion)
    elif method == "implicit":
        H = g.H(u) 
        BC_stencil(g, H, BCu)

        U = np.zeros((3, len(u)))
        U[1] += rho/dt
        U -= h[:3]

        b = rho/dt*u + S + H[3]

        u_star, info = solve_stencil(U, b)

    # "Stable fluids" method (explicit advection, implicit diffusion)
    elif method == "stable":
        A = upwind(u, BC_u=BC_u, BC_phi=BC_u, phi=u) 
        u_adv = u - dt*A

        D = mu*central_second(BC_u, N=len(u)) #diffusion
        U = -D
        U[1] += rho/dt

        b = rho/dt*u + S + D[3]

        u_star, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(U), b)

    elif method == "piso":
        return piso_step(u, p, BC_u, BC_p)
    else:
        assert(False)

    P = central_second(BC_p, N=len(p))
    b = rho/dt*central(BC_u, u_star) + P[3]

    p_next = np.zeros_like(p)
    p_next, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(P), b)
    if info != 0:
        cond_num = sparse_cond_num(stencil_to_sparse_matrix(P))
        print(f"{cond_num=}")

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
        # method = "explicit"
        # method = "implicit"
        # method = "stable"
        method = "piso"
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