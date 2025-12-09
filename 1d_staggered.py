import time
import numpy as np
import scipy as sp
import typing
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass

N = 8

R_spec = 287
T = 272 + 20
c_sound = 343

L = 1
dx = L/N
dt = 2e-2

rho = 1
ini_p = 0
ini_ux = 0

t0 = 0
t1 = 2

lam = -1.5308e-2
mu = 1.3059e-2

show_interval = 0.00
multistep = 5

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

@dataclass
class Boundary:
    type:str
    value:float

def dirichlet(val:float): return Boundary("val", val)
def neumann(val:float): return Boundary("der", val)

def BC_to_ghost(side:str, BC:Boundary, val:float) -> float:
    if BC.type == "val":
        return 2*BC.value - val
    if BC.type == "der":
        if side == "west": return -BC.value*dx + val
        if side == "east": return BC.value*dx + val
    return 0.0

# class Boundaries:
#     indices: np.ndarray
#     types: np.ndarray
#     values: np.ndarray
# 
# BOUNDARY_NONE = 0
# BOUNDARY_R_VAL = 1
# BOUNDARY_L_VAL = 2
# BOUNDARY_R_DER = 3
# BOUNDARY_L_DER = 4
# 
# def face_val_der(cells:np.ndarray, BC:Boundaries) -> typing.Tuple[np.ndarray, np.ndarray]:
#     vals = np.empty(len(cells)+1)
#     ders = np.empty(len(cells)+1)
#     vals[1:-1,] = 0.5*(cells[1:] + cells[:-1])
#     ders[1:-1] = (1/dx)*(cells[1:] + cells[:-1])
# 
#     for (i, type, B) in zip(BC.indices, BC.types, BC.values):
#         if False: None
#         elif type == BOUNDARY_R_VAL:
#             vals[i] = B
#             ders[i] = (cells[i] - B)*(2/dx)
#         elif type == BOUNDARY_L_VAL:
#             vals[i] = B
#             ders[i] = (B - cells[i-1])*(2/dx)
#         elif type == BOUNDARY_R_DER:
#             vals[i] = cells[i] - B*dx/2
#             ders[i] = B
#         elif type == BOUNDARY_L_DER:
#             vals[i] = cells[i-1] + B*dx/2
#             ders[i] = B
# 
#     return vals, ders

def cell_val(faces:np.ndarray) -> np.ndarray:
    return 0.5*(faces[1:] + faces[:-1])

# If dirichlet we do not solve in that face at all - still we do want a sensible answer to the momentum eqs
# After the predictor is done we once again enforce the boundary face so that it matches!
# This will not work for any complex boundary problem. How do we reconcile this?

# f --- g -X- h --- j

# For dirichlet we can assume that behind the wall the value is unfirom thus expand to the boundary value
# For neumann we expand according to the d

def vel_expand_apply(faces:np.ndarray, BCs) -> np.ndarray:
    out = np.empty(len(faces)+2)
    out[1:-1] = faces
    if BCs[0].type == "val":
        out[0] = BCs[0].value
        out[1] = BCs[0].value
    if BCs[0].type == "der":
        out[0] = faces[0] - BCs[0].value*dx

    if BCs[-1].type == "val":
        out[-1] = BCs[-1].value
        out[-2] = BCs[-1].value
    if BCs[-1].type == "der":
        out[-1] = faces[-1] + BCs[-1].value*dx
    return out

def vel_apply(faces:np.ndarray, BCs) -> np.ndarray:
    out = faces.copy()
    if BCs[0].type == "val":
        out[0] = BCs[0].value
    if BCs[-1].type == "val":
        out[-1] = BCs[-1].value
    return out

def vel_der(exp) -> np.ndarray:
    return (exp[2:] - exp[:-2])*(1/(2*dx))

def vel_der2(exp) -> np.ndarray:
    return np.diff(exp, n=2)*(1/dx)**2

def vel_der_upw(exp, dir_faces:np.ndarray) -> np.ndarray:
    # 0 1 2 3 4 N
    #  F F F F F
    fluxes = (exp[1:] - exp[:-1])*(1/dx)
    dir = (dir_faces >= 0)
    return dir*fluxes[:-1] + (1-dir)*fluxes[1:]

def vel_div_cell(faces:np.ndarray) -> np.ndarray:
    return (1/dx)*(faces[1:] - faces[:-1])

def face_val(cells:np.ndarray, BCs) -> np.ndarray:
    g0 = BC_to_ghost("west", BCs[ 0], cells[ 0])
    gN = BC_to_ghost("east", BCs[-1], cells[-1])
    exp = np.concatenate(([g0], cells, [gN]))
    return 0.5*(exp[1:] + exp[:-1])

def face_der(cells:np.ndarray, BCs) -> np.ndarray:
    g0 = BC_to_ghost("west", BCs[ 0], cells[ 0])
    gN = BC_to_ghost("east", BCs[-1], cells[-1])
    return np.diff(cells, prepend=g0, append=gN)*(1/dx)

def face_upw(cells:np.ndarray, vals:np.ndarray, upw_dir_dirs:np.ndarray) -> np.ndarray:
    mask = (upw_dir_dirs[1:-1] >= 0)
    upws = np.empty_like(vals)
    upws[1:-1] = mask*cells[:-1] + (mask-1)*cells[1:]
    upws[ 0] = vals[ 0] if upw_dir_dirs[ 0] >= 0 else cells[ 0]
    upws[-1] = vals[-1] if upw_dir_dirs[-1] <  0 else cells[-1]
    return upws

def der_central(faces_der:np.ndarray) -> np.ndarray:
    return 0.5*(faces_der[1:] + faces_der[:-1])

def der2_central(faces_der:np.ndarray) -> np.ndarray:
    return (1/dx)*(faces_der[1:] - faces_der[:-1])

def der_upwind(faces_upw:np.ndarray) -> np.ndarray:
    return (1/dx)*(faces_upw[1:] - faces_upw[:-1])

def matrix_free_solve(A:typing.Callable, b:np.ndarray, applyOffset:bool=True, x0:np.ndarray|None = None, rtol:float = 1e-3, maxiter:int = 200) -> typing.Tuple[np.ndarray, int]:
    Aoff = A
    Boff = b
    if applyOffset:
        offset = A(np.zeros_like(b))
        Aoff = lambda x: A(x) - offset
        Boff = b - offset

    Aop = sp.sparse.linalg.LinearOperator((len(b), len(b)), matvec=Aoff)
    return sp.sparse.linalg.cgs(Aop, Boff, x0=x0, rtol=rtol, maxiter=maxiter)

def step(un:np.ndarray, pn:np.ndarray, BCu, BCp, variant, rtol=1e-3, iters=1, alpha=1):
    # un_kn = A^-1(f_rhs - G*pn_k)
    # qn_kn = rho/dt*L^-1*D*un_kn
    # pn_kn = pn_k + qn_kn - mu/rho*D*un_kn

    S = 0
    unk = un
    pnk = pn
    for k in range(iters):
        def predA(u:np.ndarray) -> np.ndarray:
            nonlocal BCu
            nonlocal unk
            expu = vel_expand_apply(u, BCu)
            H = -rho*unk*vel_der_upw(expu, unk) + mu*vel_der2(expu)
            U = rho/dt*u - H
            return U
        
        predB = rho/dt*un + S - (face_der(pnk, BCp) if variant != "non-increment" else 0)
        predU, predIters = matrix_free_solve(predA, predB, x0=un, rtol=rtol)
        assert predIters == 0
        predU = vel_apply(predU, BCu)

        div_predU = vel_div_cell(predU)
        corrA = lambda p: der2_central(face_der(p, BCp))
        corrB = rho/dt*div_predU
        corrP, corrIters = matrix_free_solve(corrA, corrB)
        assert corrIters == 0

        u_next = vel_apply(predU - dt/rho*face_der(corrP, BCp), BCu)
        if   variant == "non-increment":    p_next = corrP
        elif variant == "increment":        p_next = pnk + corrP
        elif variant == "increment-rot":    p_next = pnk + corrP - mu/rho*div_predU

        if k == iters - 1:
            return (u_next, p_next)

        unk = (1 - alpha)*unk + alpha*u_next
        pnk = (1 - alpha)*pnk + alpha*p_next
    return None


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

        variant = "non-increment"
        # variant = "increment"
        # variant = "increment-rot"

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
