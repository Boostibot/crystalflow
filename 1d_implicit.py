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
dt = 2e-2

rho = 1
ini_p = 0
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

def apply_stencil(stencil, field, remove_diag=False, remove_RHS=False):
    assert len(stencil) == 4
    assert len(stencil[0]) == len(field)

    out = np.zeros_like(field)
    out[1:] += stencil[0, :-1]*field[:-1]
    if remove_diag == False:
        out += stencil[1]*field 
    out[:-1] += stencil[2, 1:]*field[1:]
    if remove_diag == False:
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


def central(BC, phi=None, n=N):
    A = np.zeros((4, n))
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

    #                       C                     W                           E                        C 
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
    # A = 0
    D = central_second(BC_phi, phi=phi) #diffusion
    return -rho*A + mu*D

def stencil_to_sparse_matrix(A):
    return sp.sparse.diags([A[0, :-1], A[1], A[2, 1:]], [-1, 0, 1])


# Sim Boundary types:
# Inflow
# Outflow
# Wall


# Impl boundary types (relative to a specific field):
# der +/-
# val +/-

def BC_to_ghost(side:str, BC:Boundary, val:float) -> float:
    if BC.type == "val":
        return 2*BC.value - val
    if BC.type == "der":
        if side == "west": return -BC.value*dx + val
        if side == "east": return BC.value*dx + val
    return 0.0

def ghost_to_BC(side:str, BC_kind:str, val:float, ghost:float) -> Boundary:
    if BC_kind == "val":
        return Boundary(BC_kind, (val + ghost)/2)
    if BC_kind == "der":
        if side == "west": return Boundary(BC_kind, (val - ghost)/dx)
        if side == "east": return Boundary(BC_kind, (ghost - val)/dx)
    return Boundary("none")

def BC_convert(side:str, BC_kind:str, BC:Boundary, val:float) -> Boundary:
    if BC_kind == BC.type:
        return BC
    ghost = BC_to_ghost(side, BC, val)
    return ghost_to_BC(side, BC_kind, val, ghost)

def expand(BCs, phi):
    out = np.empty(phi.shape[0]+2, dtype=phi.dtype)
    out[1: -1] = phi
    out[0] = BC_to_ghost("west", BCs[0], phi[0])
    out[-1] = BC_to_ghost("east", BCs[-1], phi[-1])
    return out

def contract(phi):
    return phi[1:-1]

BC_neumann = (neumann(), neumann())
BC_dirichlet = (dirichlet(), dirichlet())
BC_none = (Boundary("none"), Boundary("none"))

def pder_boundary_west(BC_u, u:np.ndarray, u_dt:float) -> Boundary:
    u_exp = expand(BC_u, u)
    u0, u1, u2 = u_exp[0], u_exp[1], u_exp[2]
    uf = (u0 + u1)/2
    ug = (u1 + u2)/2
    dx_uf = (u1 - u0)/dx
    dx_ug = (u2 - u1)/dx
    # dxx_ufg = 2*(dx_ug - dx_uf)/dx
    dxx_ufg = 0
    dt_uf = u_dt
    dt_ug = u_dt
    Sf = 0
    Sg = 0
    p_der = -rho*(dt_uf + uf*dx_uf) + mu*dxx_ufg + Sf

    # higher p_der|f
    # => (explicit scheme) dt_u = terms - (p_der|f - p_der|g) => lower dt_u
    # => lower u1
    # => lower dx_uf
    # => higher p_der

    return neumann(p_der)

def pder_boundary(side:str, u0:float, u1:float, u2:float, u_dt:float = 0) -> float:
    uf = (u0 + u1)/2
    ug = (u1 + u2)/2
    dx_uf = (u1 - u0)/dx
    dx_ug = (u2 - u1)/dx
    dxx_ufg = (dx_ug - dx_uf)/dx
    dxx_ufg = 0
    dt_uf = u_dt
    dt_ug = u_dt
    Sf = 0
    Sg = 0

    if side == "west":
        p_der = -rho*(dt_uf + uf*dx_uf) + mu*dxx_ufg + Sf
    else:
        p_der = -rho*(dt_ug + ug*dx_ug) + mu*dxx_ufg + Sg
    return p_der

def calc_p_boundaries(wtype, etype, BC_u, u:np.ndarray, p:np.ndarray):
    u_exp = expand(BC_u, u)
    dp_0 = pder_boundary("west", u_exp[ 0], u_exp[ 1], u_exp[ 2])
    dp_N = pder_boundary("east", u_exp[-3], u_exp[-2], u_exp[-1])

    BC_0 = BC_convert("west", wtype, neumann(dp_0), p[0])
    BC_N = BC_convert("east", etype, neumann(dp_N), p[-1])
    return (BC_0, BC_N)

def piso_step(u, p, BC_u, BC_p, piso_iters=2, relax=1, BC_source=BC_neumann):
    # predictor
    h = H(u, BC_u=BC_u, BC_phi=BC_u)
    A = h[1]
    B = h[3]
    S = 0
    alpha = rho/dt
    R = alpha*u + S

    A_pred = -h
    A_pred[1] += alpha 
    b_pred = R - central(BC_p, p) + B

    u_pred, info = sp.sparse.linalg.cgs(stencil_to_sparse_matrix(A_pred), b_pred)
    if info != 0:
        cond_num = sparse_cond_num(stencil_to_sparse_matrix(P))
        print(f"{cond_num=}")
    assert info == 0


    u_star = u_pred
    p_star = p
    # corrector loop
    for k in range(piso_iters):
        L = apply_stencil(h, u_star) + R

        # Expand according to predictor equation and apply
        p_star_exp = expand(BC_p, p_star)
        u_star_exp = expand(BC_u, u_star)
        L_exp = expand(BC_u, (alpha - A)*u_star)
        L_exp[0] += (p_star_exp[1] - p_star_exp[0])/dx
        L_exp[-1] += (p_star_exp[-1] - p_star_exp[-2])/dx
        L_exp[1:-1] = L
        div_L_exp = central(BC_none, L_exp, n=L_exp.shape[0])
        div_L = contract(div_L_exp)

        # posisson EQ for pressure
        P = central_second(BC_p)
        p_star_next, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(P), div_L)
        if info != 0:
            cond_num = sparse_cond_num(stencil_to_sparse_matrix(P))
            print(f"{cond_num=}")
        assert info == 0

        # explicit update of velocity
        p_next_div = central(BC_p, p_star_next)
        u_star_next = (L - p_next_div)/(alpha - A)

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

def step(u, p, BC_u, BC_p, method="implicit", increment=False):
    S = 0

    u_star = None
    u_strange = None

    # Classic projection method (explicit advection diffusion)
    if method == "explicit":
        u_star = u + dt/rho*( 
            H(u, BC_u, BC_u, phi=u) 
            + S 
            - (central(BC_p, p) if increment else 0)
        ) 
        
    # Implicit projection method (implicit advection diffusion)
    elif method == "implicit":
        h = H(u, BC_u, BC_u)

        U = np.zeros((3, len(u)))
        U[1] += rho/dt
        U -= h[:3]

        b = rho/dt*u + S + h[3] - (central(BC_p, p) if increment else 0)

        u_star, info = sp.sparse.linalg.cgs(stencil_to_sparse_matrix(U), b)

    # "Stable fluids" method (explicit advection, implicit diffusion)
    elif method == "split":
        A = upwind(u, BC_u=BC_u, BC_phi=BC_u, phi=u) 
        u_adv = u + dt*A

        D = mu*central_second(BC_u) #diffusion
        U = -D
        U[1] += rho/dt

        b = rho/dt*u_adv + S + D[3] - (central(BC_p, p) if increment else 0)

        u_star, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(U), b)

    elif method == "piso":
        return piso_step(u, p, BC_u, BC_p)
    else:
        assert(False)

    P = central_second(BC_p)
    b = rho/dt*central(BC_u, phi=u_star) + P[3]
    
    # div(p^n+1 - p^n) = -1/dt(u^n+1 - u*)
    # p_delta = p^n+1 - p^n
    # => DIV: lap(p_delta) = 1/dt*div(u*)
    # => NEX: u^n+1 = u* - dt*div(p_delta)
    #         p^n+1 = p^n + p_delta

    p_corr, info = sp.sparse.linalg.cg(stencil_to_sparse_matrix(P), b)
    if info != 0:
        cond_num = sparse_cond_num(stencil_to_sparse_matrix(P))
        print(f"{cond_num=}")
        assert info == 0

    u_next = u_star - dt/rho*central(BC_p, p_corr)
    p_next = p + p_corr if increment else p_corr
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
    inflow_ux_last = 0
    while t < t1:
        inflow_ux = min(1, t)
        inflow_ux_dt = (inflow_ux - inflow_ux_last)/dt
        inflow_ux_last = inflow_ux
        # method = "explicit"
        method = "implicit"
        # method = "split"
        # method = "piso"
        increment = True
        # increment = False
        BC_u = (dirichlet(inflow_ux), neumann(0))
        # BC_p = (pder_boundary_west(BC_u, u, inflow_ux_dt), dirichlet(0))
        BC_p = (neumann(0),           dirichlet(0))

        u_next, p_next = step(u, p, BC_u, BC_p, method=method, increment=increment)

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