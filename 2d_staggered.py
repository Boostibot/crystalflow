import time
import numpy as np
import scipy as sp
import typing
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass

Nx = 20
Ny = 20

R_spec = 287
T = 272 + 20
c_sound = 343

Lx = 1
Ly = 1
dx = Lx/Nx
dy = Ly/Ny
dt = 2e-2

ini_p = 0
ini_ux = 0

t0 = 0
t1 = 2

nu = 1.3059e-2

show_interval = 0.00
multistep = 5

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

BOUND_NONE = 0xffffffff

# Bits 
#  [0]: direction negative/positive
#  [1]: direction normal/tangential
#  [2]: boundary kind value/derivative (dirichlet/neumann)
BOUND_VAL_W = 0b000 
BOUND_VAL_E = 0b001
BOUND_VAL_S = 0b010
BOUND_VAL_N = 0b011

BOUND_DER_W = 0b100 
BOUND_DER_E = 0b101
BOUND_DER_S = 0b110
BOUND_DER_N = 0b111

def bound_type_transpose(types:np.ndarray) -> np.ndarray:
    return np.bitwise_xor(types, 0b010)

@dataclass
class Boundaries:
    values:np.ndarray
    types:np.ndarray
    indices:np.ndarray

class Grid:
    ux_bc:Boundaries
    uy_bc:Boundaries
    p_bc:Boundaries

def contract(f:np.ndarray) -> np.ndarray:
    return f[1:-1, 1:-1] 

def expand(f:np.ndarray) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f
    return out

def expand_velx_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f

    for val, type, (x, y) in zip(bcs.values, bcs.types, bcs.indices + 1):
        if False: None
        elif type == BOUND_VAL_W:
            out[x, y] = val
            out[x-1, y] = val
        elif type == BOUND_VAL_E:
            out[x, y] = val
            out[x+1, y] = val
        elif type == BOUND_VAL_S: out[x, y-1] = 2*val - out[x, y]
        elif type == BOUND_VAL_N: out[x, y+1] = 2*val - out[x, y]
        elif type == BOUND_DER_W: out[x-1, y] = out[x, y] - val*dx
        elif type == BOUND_DER_E: out[x+1, y] = out[x, y] + val*dx
        elif type == BOUND_DER_S: out[x, y-1] = out[x, y] - val*dy
        elif type == BOUND_DER_N: out[x, y+1] = out[x, y] + val*dy

    return out

def expand_vely_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f

    for val, type, (x, y) in zip(bcs.values, bcs.types, bcs.indices + 1):
        if False: None
        elif type == BOUND_VAL_W: out[x-1, y] = 2*val - out[x, y]
        elif type == BOUND_VAL_E: out[x+1, y] = 2*val - out[x, y]
        elif type == BOUND_VAL_S:
            out[x, y] = val
            out[x, y-1] = val
        elif type == BOUND_VAL_N:
            out[x, y] = val
            out[x, y+1] = val
        elif type == BOUND_DER_W: out[x-1, y] = out[x, y] - val*dx
        elif type == BOUND_DER_E: out[x+1, y] = out[x, y] + val*dx
        elif type == BOUND_DER_S: out[x, y-1] = out[x, y] - val*dy
        elif type == BOUND_DER_N: out[x, y+1] = out[x, y] + val*dy

    return out

def apply_velx_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    flat_indices = np.where(bcs.types == BOUND_VAL_W or bcs.types == BOUND_VAL_E)
    f[bcs.indices[flat_indices]] = bcs.values[flat_indices]
    return f

def apply_vely_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    flat_indices = np.where(bcs.types == BOUND_VAL_N or bcs.types == BOUND_VAL_S)
    f[bcs.indices[flat_indices]] = bcs.values[flat_indices]
    return f

def expand_vels_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    return np.array((expand_velx_bcs(f[0], bcs[0]), expand_vely_bcs(f[1], bcs[1])))

def apply_vels_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    apply_velx_bcs(f[0], bcs[0])
    apply_vely_bcs(f[1], bcs[1])
    return f

expand_vel_bcs = (expand_velx_bcs, expand_vely_bcs)
apply_vel_bcs = (apply_velx_bcs, apply_vely_bcs)

def expand_cell_bcs(f:np.ndarray, bcs:Boundaries) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f

    for val, type, (x, y) in zip(bcs.values, bcs.types, bcs.indices + 1):
        if False: None
        elif type == BOUND_VAL_W: out[x-1, y] = 2*val - out[x, y]
        elif type == BOUND_VAL_E: out[x+1, y] = 2*val - out[x, y]
        elif type == BOUND_VAL_S: out[x, y-1] = 2*val - out[x, y]
        elif type == BOUND_VAL_N: out[x, y+1] = 2*val - out[x, y]
        elif type == BOUND_DER_W: out[x-1, y] = out[x, y] - val*dx
        elif type == BOUND_DER_E: out[x+1, y] = out[x, y] + val*dx
        elif type == BOUND_DER_S: out[x, y-1] = out[x, y] - val*dy
        elif type == BOUND_DER_N: out[x, y+1] = out[x, y] + val*dy

    return out

def Dx(f:np.ndarray) -> np.ndarray: return (f[2:, 1:-1] - f[:-2, 1:-1])/(2*dx)
def Dy(f:np.ndarray) -> np.ndarray: return (f[1:-1, 2:] - f[1:-1, :-2])/(2*dy)

def Dpx(f:np.ndarray) -> np.ndarray: return (f[2:  , 1:-1] - f[1:-1, 1:-1])/(dx)
def Dnx(f:np.ndarray) -> np.ndarray: return (f[1:-1, 1:-1] - f[ :-2, 1:-1])/(dx)

def Dpy(f:np.ndarray) -> np.ndarray: return (f[1:-1, 2:  ] - f[1:-1, 1:-1])/(dy)
def Dny(f:np.ndarray) -> np.ndarray: return (f[1:-1, 1:-1] - f[1:-1,  :-2])/(dy)

def DDx(f:np.ndarray) -> np.ndarray:
    return (f[2:, 1:-1] - f[1:-1, 1:-1] + f[:-2, 1:-1])/dx**2

def DDy(f:np.ndarray) -> np.ndarray:
    return (f[1:-1, 2:] - f[1:-1, 1:-1] + f[1:-1, :-2])/dy**2

D = (Dx, Dy)
DD = (DDx, DDy)
Dp = (Dpx, Dpy)
Dn = (Dnx, Dny)

def Lap(exp:np.ndarray)   -> np.ndarray: return DDx(exp) + DDy(exp)
def Div(exp:np.ndarray)   -> np.ndarray: return Dx(exp[0]) + Dy(exp[1])
def Divp(exp:np.ndarray)  -> np.ndarray: return Dpx(exp[0]) + Dpy(exp[1])
def Divn(exp:np.ndarray)  -> np.ndarray: return Dnx(exp[0]) + Dny(exp[1])
def Grad(exp:np.ndarray)  -> np.ndarray: return np.array((Dx(exp), Dy(exp)))
def Gradp(exp:np.ndarray) -> np.ndarray: return np.array((Dpx(exp), Dpy(exp)))
def Gradn(exp:np.ndarray) -> np.ndarray: return np.array((Dnx(exp), Dny(exp)))

def matrix_free_solve(A:typing.Callable, b:np.ndarray, expandOffset:bool=True, x0:np.ndarray|None = None, rtol:float = 1e-3, maxiter:int = 200) -> typing.Tuple[np.ndarray, int]:
    Aoff = A
    Boff = b
    if expandOffset:
        offset = A(np.zeros_like(b))
        Aoff = lambda x: A(x) - offset
        Boff = b - offset

    Aop = sp.sparse.linalg.LinearOperator((len(b), len(b)), matvec=Aoff)
    return sp.sparse.linalg.cgs(Aop, Boff, x0=x0, rtol=rtol, maxiter=maxiter)

def step(un:np.ndarray, pn:np.ndarray, u_bcs:Boundaries, p_bc:Boundaries, variant, rtol=1e-3, iters=1):
    # un_kn = A^-1(f_rhs - G*pn_k)
    # qn_kn = rho/dt*L^-1*D*un_kn
    # pn_kn = pn_k + qn_kn - mu/rho*D*un_kn

    S = 0
    unk = un
    pnk = pn #todo guess
    for k in range(iters):
        predU = np.empty_like(unk)
        for i in range(2):
            def predA(u:np.ndarray) -> np.ndarray:
                nonlocal i
                nonlocal unk
                nonlocal u_bcs

                ue = expand_vel_bcs[i](u, u_bcs[i])
                U = 1/dt*u + unk*Grad(ue) - nu*Lap(ue)
                return apply_vel_bcs[i](U, u_bcs[i])
            
            pGrad = Dp[i](expand_cell_bcs(pnk, p_bc)) if variant != "non-increment" else 0
            predB = 1/dt*un[i] + S - pGrad
            predU[i], predIters = matrix_free_solve(predA, predB, x0=un[i], rtol=rtol)
            assert predIters == 0
        apply_vels_bcs(predU)

        div_predU = Divn(expand_vels_bcs(predU, u_bcs))
        def corrA(p:np.ndarray):
            nonlocal p_bc
            return Lap(expand_cell_bcs(p, p_bc))
        
        corrB = 1/dt*div_predU
        corrP, corrIters = matrix_free_solve(corrA, corrB)
        assert corrIters == 0

        u_next = predU - dt*Gradp(expand_cell_bcs(corrP, p_bc))
        apply_vels_bcs(u_next, u_bcs)

        if   variant == "non-increment":    p_next = corrP
        elif variant == "increment":        p_next = pnk + corrP
        elif variant == "increment-rot":    p_next = pnk + corrP - nu*div_predU

        unk = u_next
        pnk = p_next

    # todo filter
    return (unk, pnk)


def graph():
    global inflow_ux

    # Setup
    cell_centers = (np.arange(N) + 0.5)*(L/N)
    face_positions = np.arange(N+1)*(L/N)

    p = np.zeros(N) + ini_p
    u = np.zeros(N+1) + ini_ux

    plt.ion()  # Turn on interactive mode
    # plt.tight_layout()

    fig, (ax_ux, ax_p, ax_stats) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    u_cells, = ax_ux.plot(face_positions, u, label='u')
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

    ax_text = ax_stats.text(0, 0.5, "", fontsize=12, ha='left', va='center')
    ax_stats.axis('off')

    SMOOTH_N = 10
    alpha = 0.3
    simple_loops = 1
    simple_alpha = 1
    iter = 0
    t = dt

    t_arr = []
    div_u_arr = []
    lap_p_arr = []
    while t <= t1:
        inflow_ux = min(1, t)

        # variant = "non-increment"
        # variant = "increment"
        variant = "increment-rot"

        # if iter % SMOOTH == 0: variant = "non-increment"
        # else:                  variant = "increment"
        # else:                  variant = "increment-rot"
        BC_u = (dirichlet(inflow_ux), neumann(0))
        BC_p = (neumann(0),           dirichlet(0))

        u_next, p_next = step(u, p, BC_u, BC_p, variant=variant, rtol=1e-2, iters=simple_loops, alpha=simple_alpha)

        # blended
        # u_next_2, p_next_2 = step(u, p, BC_u, BC_p, variant="non-increment", rtol=1e-2)
        # u_next = (1 - alpha)*u_next + alpha*u_next_2
        # p_next = (1 - alpha)*p_next + alpha*p_next_2

        u, u_next = u_next, u
        p, p_next = p_next, p

        div_u = np.linalg.norm(der_central(face_der(u, BC_u)))
        lap_p = np.linalg.norm(der2_central(face_der(p, BC_p)))
        t_arr.append(t)
        div_u_arr.append(div_u)
        lap_p_arr.append(lap_p)
    
        if iter % multistep == 0:
            cfl = get_cfl(u)
            Ma = get_mach_number(u)
            stats = \
                f"""
                t = {float(t):.6} 
                div(u) = {div_u:.4e} 
                lap(p) = {lap_p:.4e} 
                CFL = {cfl:.2} 
                Ma = {Ma:.2}
                """
            
            u_cells.set_ydata(u)
            p_cells.set_ydata(p)
            ax_ux.set_title(f"variant = {variant} iters = {simple_loops} t = {float(t):.6}")
            ax_text.set_text(stats)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # plt.pause(show_interval)

        t += dt
        iter += 1

    ax_stats.clear()
    ax_stats.plot(t_arr, div_u_arr, label='div_u')
    ax_stats.plot(t_arr, lap_p_arr, label='lap_p')
    ax_stats.axis('on')
    ax_stats.set_yscale('log')

    ax_stats.minorticks_on()
    ax_stats.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=10))
    ax_stats.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=[2, 5], numticks=100))
    ax_stats.grid(which='major', linewidth=0.9)
    ax_stats.grid(which='minor', linewidth=0.5, linestyle=':')

    ax_stats.legend()

    plt.ioff()  # Turn off interactive mode after loop ends
    plt.show()

graph()
