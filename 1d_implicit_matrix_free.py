import time
import numpy as np
import scipy as sp
import typing
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass

N = 100

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

def face_val(cells:np.ndarray, BCs) -> np.ndarray:
    g0 = BC_to_ghost("west", BCs[ 0], cells[ 0])
    gN = BC_to_ghost("east", BCs[-1], cells[-1])
    exp = np.concatenate(([g0], cells, [gN]))
    return 0.5*(exp[1:] + exp[:-1])

def face_der(cells:np.ndarray, BCs) -> np.ndarray:
    g0 = BC_to_ghost("west", BCs[ 0], cells[ 0])
    gN = BC_to_ghost("east", BCs[-1], cells[-1])
    return np.diff(cells, prepend=g0, append=gN)*(1/dx)

def face_upw(cells:np.ndarray, vals:np.ndarray, upw_dir_vals:np.ndarray) -> np.ndarray:
    mask = (upw_dir_vals[1:-1] >= 0)
    upws = np.empty_like(vals)
    upws[1:-1] = mask*cells[:-1] + (mask-1)*cells[1:]
    upws[ 0] = vals[ 0] if upw_dir_vals[ 0] >= 0 else cells[ 0]
    upws[-1] = vals[-1] if upw_dir_vals[-1] <  0 else cells[-1]
    return upws

def der_central(faces_der:np.ndarray) -> np.ndarray:
    return 0.5*(faces_der[1:] + faces_der[:-1])

def der2_central(faces_der:np.ndarray) -> np.ndarray:
    return (1/dx)*(faces_der[1:] - faces_der[:-1])

def der_upwind(faces_upw:np.ndarray) -> np.ndarray:
    return (1/dx)*(faces_upw[1:] - faces_upw[:-1])

def BC_to_face_val_der_upw(side:bool, BC:Boundary, cell:float) -> typing.Tuple[float, float, float]:
    if BC.type == "val":
        val = BC.value
        der = (cell - val)/(dx/2) if side else (val - cell)/(dx/2)
    if BC.type == "der":
        der = BC.value
        val = cell - der*dx/2 if side else cell + der*dx/2
    return val, der

def face_val_der_upw(cells:np.ndarray, BCs) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(cells)

    g0 = BC_to_ghost("west", BCs[ 0], cells[ 0])
    gN = BC_to_ghost("east", BCs[-1], cells[-1])
    exp = np.concatenate([g0], cells, [gN])

    vals = np.empty(N+1)
    ders = np.empty(N+1)

    vals[1:-1] = 0.5*(cells[1:] + cells[:-1])
    ders[1:-1] = (1/dx)*(cells[1:] - cells[:-1])

    vals[ 0], ders[ 0] = BC_to_face_val_der_upw(True, BCs[ 0], cells[ 0])
    vals[-1], ders[-1] = BC_to_face_val_der_upw(False, BCs[-1], cells[-1])
    return vals, ders


def matrix_free_solve(A:typing.Callable, b:np.ndarray, applyOffset:bool=True, x0:np.ndarray|None = None, rtol:float = 1e-3, maxiter:int = 200) -> typing.Tuple[np.ndarray, int]:
    Aoff = A
    Boff = b
    if applyOffset:
        offset = A(np.zeros_like(b))
        Aoff = lambda x: A(x) - offset
        Boff = b - offset

    Aop = sp.sparse.linalg.LinearOperator((len(b), len(b)), matvec=Aoff)
    return sp.sparse.linalg.cgs(Aop, Boff, x0=x0, rtol=rtol, maxiter=maxiter)

# def corrADiscrete(p:np.ndarray) -> np.ndarray:
#     nonlocal BCp
#     nonlocal BCpe
#     exp = np.empty(len(p)+4)
#     exp[2:-2] = p
#     exp[1] = exp[2] - dx*BCp[0].value
#     exp[0] = exp[1] - dx*BCp[0].value

#     exp[-2] = 2*BCp[1].value - exp[-3]
#     exp[-1] = exp[-2]

#     d = 1/(4*dx*dx)
#     return d*(exp[4:] - 2*exp[2:-2] + exp[:-4])
#     # der = der_central(face_der(p, BCp))
#     # der2 = der_central(face_der(der, BCpe))
#     # return der2

def step(un:np.ndarray, pn:np.ndarray, BCu, BCp, variant, rtol=1e-3, outer_iters=1, inner_iters=1, alpha=1, smoothing="none"):
    # un_kn* = A^-1(f_rhs - G*pn_k)
    # qn_kn = L^-1(rho/dt*D*un_kn*)
    # pn_kn* = pn_k* + qn_kn - mu/rho*D*un_kn*

    # un_kn = un_kn* - dt*D*qn_kn
    # -- pn_kn = D^-1() ... so that it matches the eqs (implcit)

    # un_kn** = A^-1(f_rhs +- G*pn_k) = un_kn* + A^-1*G*pn_k 
    # pn_kn = L^-1(rho/dt*D*un_kn**) = L^-1(rho/dt * D * (un_kn + A^-1*G*pn_k))
    #       ~ L^-1(rho/dt * D * A^-1*G*pn_k)
    #       ~ L^-1(rho/dt * D * A^-1*f_rhs)

    # pn_k^ ... prediction of next step
    # pn_kn* ... next step
    # pn_ke  = pn_kn* - pn_kn^

    # pressure -> velocity -> pressure
    #    |                       |
    #    |_______________________|


    #         exact      smoothed
    # pressure -> velocity -> pressure -> velocity
    #                |                       |
    #                |_______________________|
    # p^ = (I-s*LL)^-1 p

    # un_kn* = A^-1(f_rhs - G*(pn_kn + pn_ke))
    # un_kn*^ = A^-1(f_rhs - G*(pn_kn^ + pn_ke))
    # un_kn*e = un_kn* - un_kn*^ 
    #         = A^-1(G*(pn_kn - pn_kn^))
    #  
    # qn_kn  = L^-1(rho/dt*D*un_kn*)
    # qn_kn^ = L^-1(rho/dt*D*un_kn*^)
    # qn_kne = qn_kn - qn_kn^ = L^-1(A^-1(G*(pn_kn - pn_kn^)))
    # 
    # pn_kn* = pn_k^ + pn_ke + qn_kn - mu/rho*D*un_kn*

    # un_kn* = A^-1(f_rhs - G*pn_k)
    # qn_kn = L^-1(rho/dt*D*un_kn*)
    # pn_kn* = pn_k* + qn_kn - mu/rho*D*un_kn*

    I_kernel = np.array([0, 0, 1, 0, 0])
    d4_kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
    smooth_kernel = I_kernel - dt*d4_kernel

    S = 0
    unk = un
    pnk = pn
    corrP = None
    for k in range(outer_iters):
        unk_f_val = face_val(unk, BCu)
        def predA(u:np.ndarray) -> np.ndarray:
            nonlocal BCu
            nonlocal unk
            nonlocal unk_f_val
            
            uf_val = face_val(u, BCu)
            uf_der = face_der(u, BCu)
            uf_upw = face_upw(u, uf_val, unk_f_val)
            H = -rho*unk*der_upwind(uf_upw) + mu*der2_central(uf_der)
            U = rho/dt*u - H
            return U
        
        predB = rho/dt*un + S - (der_central(face_der(pnk, BCp)) if variant != "non-increment" else 0)
        predU, predIters = matrix_free_solve(predA, predB, x0=un, rtol=rtol)
        assert predIters == 0

        div_predU = der_central(face_der(predU, BCu))
        corrA = lambda p: der2_central(face_der(p, BCp))
        corrB = rho/dt*div_predU
        corrP, corrIters = matrix_free_solve(corrA, corrB)
        assert corrIters == 0

        unk = predU
        if   variant == "non-increment":    pnk = corrP
        elif variant == "increment":        pnk = pnk + corrP
        elif variant == "increment-rot":    pnk = pnk + corrP 
        elif variant == "increment-damp":   pnk = pnk + corrP - 0.5*dt*mu/rho*der2_central(face_der(corrP, BCp))

        if smoothing == "explicit":
            exp = np.concatenate(([pnk[0], pnk[0]], pnk, [pnk[-1], pnk[-1]]))
            pnk = np.convolve(exp, smooth_kernel, 'valid')

        if variant == "increment-rot":   pnk -= mu/rho*div_predU

        # un_kns = unk + dt/rho*der_central(face_der(pnk, BCp))
        # smoothB = rho/dt*der_central(face_der(un_kns, BCu))
        # pn_kns, smoothIters = matrix_free_solve(corrA, smoothB, rtol=rtol)
        # assert smoothIters == 0
        # pnk = pn_kns

    u_out = unk - dt/rho*der_central(face_der(corrP, BCp))
    p_out = pnk

    # Smoothing 
    # smooth_factor = 1e-6
    # I_kernel = np.array([0, 0, 1, 0, 0])
    # d4_kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / dx**4
    # smooth_kernel = I_kernel - smooth_factor*dt*d4_kernel
    # def smoothA(p):
    #     nonlocal smooth_kernel
    #     exp = np.concatenate(([p[0], p[0]], p, [p[-1], p[-1]]))
    #     return np.convolve(exp, smooth_kernel, 'valid')
    
    # smoothP, smoothIters = matrix_free_solve(smoothA, pnk, x0=pnk, rtol=rtol)
    # assert smoothIters == 0
    # p_out = smoothP

    # Smoothing from method
    # smoothB = der_central(face_der(pnk, BCp))
    # # smoothB = rho/dt*un + S
    # un_kns, smoothIters = matrix_free_solve(predA, smoothB, rtol=rtol)
    # assert smoothIters == 0
    # un_kns += u_out

    if smoothing == "non-increment":
        un_kns = u_out + dt/rho*der_central(face_der(pnk, BCp))
        smoothB = rho/dt*der_central(face_der(un_kns, BCu))
        pn_kns, smoothIters = matrix_free_solve(corrA, smoothB, rtol=rtol)
        assert smoothIters == 0
        p_out = pn_kns

    # un_kn** = A^-1(f_rhs +- G*pn_k) = un_kn* + A^-1*G*pn_k 
    # pn_kn = L^-1(rho/dt*D*un_kn**)

    # Rhie chow
    # u_out_fder = face_der(u_out, BCu); 
    # smoothA = lambda p: der_central(face_der(p, BCp))
    # smoothB = (
    #     - rho*(u_out - un)/dt 
    #     + mu*der2_central(u_out_fder) 
    #     - rho*u_out*der_central(u_out_fder) 
    #     + S
    # )
    # p_out, smothIters = matrix_free_solve(smoothA, smoothB, x0=pnk, rtol=rtol)
    # assert smothIters == 0

    return u_out, p_out

def graph():
    global inflow_ux

    # Setup
    cell_centers = (np.arange(N) + 0.5)*(L/N)
    face_positions = np.arange(N+1)*(L/N)

    p = np.zeros(N) + ini_p
    u = np.zeros(N) + ini_ux

    plt.ion()  # Turn on interactive mode
    # plt.tight_layout()

    fig, (ax_ux, ax_p, ax_stats) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    u_cells, = ax_ux.plot(cell_centers, u, label='u')
    # u_faces, = ax_ux.plot(face_positions, faces_avg_ux, 's', label='ux faces')

    # ax_ux.set_xlim(0, 1)
    # ax_ux.set_xlim(0, 0.25)
    ax_ux.set_ylim(0, 2)
    ax_ux.set_xlabel('x')
    ax_ux.set_ylabel('ux')
    ax_ux.legend()
    ax_ux.grid(True)

    p_cells, = ax_p.plot(cell_centers, p, label='p')
    # p_faces, = ax_p.plot(face_positions, faces_avg_p, 's', label='ro faces')

    # ax_p.set_xlim(0, 1)
    # ax_p.set_xlim(0, 0.25)
    ax_p.set_ylim(0, 2)
    ax_p.set_xlabel('x')
    ax_p.set_ylabel('p')
    ax_p.legend()
    ax_p.grid(True)

    ax_text = ax_stats.text(0, 0.5, "", fontsize=12, ha='left', va='center')
    ax_stats.axis('off')

    outer_iters = 6
    inner_iters = 1
    iter = 0
    t = 0

    t_arr = []
    div_u_arr = []
    lap_p_arr = []
    while t <= t1:
        inflow_ux = min(1, t)

        # smoothing = "none"
        smoothing = "non-increment"
        # smoothing = "explicit"

        # variant = "non-increment"
        # variant = "increment"
        variant = "increment-rot"

        BC_u = (dirichlet(inflow_ux), neumann(0))
        BC_p = (neumann(0),           dirichlet(0))

        u_next, p_next = step(u, p, BC_u, BC_p, variant=variant, rtol=1e-2, outer_iters=outer_iters, inner_iters=inner_iters, smoothing=smoothing)

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
            ax_ux.set_title(f"variant = {variant} iters_out = {outer_iters} iters_in = {inner_iters} t = {float(t):.6}")
            ax_text.set_text(stats)

            fig.canvas.draw()
            fig.canvas.flush_events()

            if show_interval > 0:
                plt.pause(show_interval)

        t += dt
        iter += 1

    ax_stats.clear()
    ax_stats.plot(t_arr, div_u_arr, label='div_u')
    ax_stats.plot(t_arr, np.array(lap_p_arr)*dx, label='lap_p')
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
