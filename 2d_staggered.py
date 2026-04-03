import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, List, Dict, Literal, Callable, Iterable, Union

from platform import system
def plt_maximize():
    # See discussion: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "Windows":
            cfm.window.state("zoomed")  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == "QT4Agg":
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        raise RuntimeError("plt_maximize() is not implemented for current backend:", backend)

R_spec = 287
T = 272 + 20
c_sound = 343

# Predeclared globals. The values are set in the main function
nx, ny, nu, rho, Lx, Ly, dx, dy, dt = 0, 0, 0, 0, 0, 0, 0, 0, 0
rho = 1

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

Side         = Literal["N", "S", "E", "W"]
LowBoundType = Literal["val", "der"]
BoundaryType = Literal["inflow", "outflow", "wall"]

@dataclass
class LowBound: # Low level boundary for a field
    side:Side
    type:LowBoundType
    xs:int|np.ndarray
    ys:int|np.ndarray
    value:float|np.ndarray
    
    @staticmethod
    def concat(a:Union['LowBound', None], b: Union['LowBound', None]) -> 'LowBound':
        assert a or b
        if a is None: return b
        if b is None: return a

        xs = np.concatenate([a.xs, b.xs])
        ys = np.concatenate([a.ys, b.ys])
        value = np.concatenate([a.value, b.value])
        return LowBound(a.side, a.type, xs, ys, value)
    
LowBounds = Dict[str, LowBound] #collection of LowBounds by type

@dataclass
class Boundary: #High level boundary for the whole simulation
    side:Side
    type:BoundaryType
    xs:np.ndarray
    ys:np.ndarray
    valp:np.ndarray
    valu:np.ndarray
    valv:np.ndarray

    def __init__(self, side:Side, type:BoundaryType, xs:int|np.ndarray, ys:int|np.ndarray, valp:float|np.ndarray, valu:float|np.ndarray, valv:float|np.ndarray):
        if   isinstance(xs, np.ndarray):   l = len(xs)
        elif isinstance(ys, np.ndarray):   l = len(ys)
        elif isinstance(valp, np.ndarray): l = len(valp)
        elif isinstance(valu, np.ndarray): l = len(valu)
        elif isinstance(valv, np.ndarray): l = len(valv)
        else: l = 1

        self.side = side
        self.type = type
        self.xs = xs if isinstance(xs, np.ndarray) else np.full(l, xs, dtype=int)
        self.ys = ys if isinstance(ys, np.ndarray) else np.full(l, ys, dtype=int)
        self.valp = valp if isinstance(valp, np.ndarray) else np.full(l, valp, dtype=float)
        self.valu = valu if isinstance(valu, np.ndarray) else np.full(l, valu, dtype=float)
        self.valv = valv if isinstance(valv, np.ndarray) else np.full(l, valv, dtype=float)

    @staticmethod
    def inflow(side:Side, xs:int|np.ndarray, ys:int|np.ndarray, velx:float|np.ndarray, vely:float|np.ndarray) -> 'Boundary':
        return Boundary(side, "inflow", xs, ys, 0.0, velx, vely)
    
    @staticmethod
    def outflow(side:Side, xs:int|np.ndarray, ys:int|np.ndarray) -> 'Boundary':
        return Boundary(side, "outflow", xs, ys, 0.0, 0.0, 0.0)
    
    @staticmethod
    def noslip(side:Side, xs:int|np.ndarray, ys:int|np.ndarray) -> 'Boundary':
        return Boundary(side, "noslip", xs, ys, 0.0, 0.0, 0.0)
        
    @staticmethod
    def slip(side:Side, xs:int|np.ndarray, ys:int|np.ndarray) -> 'Boundary':
        return Boundary(side, "slip", xs, ys, 0.0, 0.0, 0.0)
    
    @staticmethod
    def concat(a:Union['Boundary', None], b: Union['Boundary', None]) -> 'Boundary':
        assert a or b
        if a is None: return b
        if b is None: return a

        xs = np.concatenate([a.xs, b.xs])
        ys = np.concatenate([a.ys, b.ys])
        valp = np.concatenate([a.valp, b.valp])
        valu = np.concatenate([a.valu, b.valu])
        valv = np.concatenate([a.valv, b.valv])
        return Boundary(a.side, a.type, xs, ys, valp, valu, valv)
    
    @staticmethod
    def to_dict(bounds:Iterable['Boundary']) -> Dict[str, 'Boundary']:
        out:Dict['Boundary'] = dict()
        for b in bounds:
            name = b.type + b.side
            out[name] = Boundary.concat(out.get(name), b)
        return out
    
    @staticmethod
    def to_low_bounds(bounds:Iterable['Boundary'], dictify=True) -> Tuple[LowBounds, LowBounds, LowBounds]:
        pbs = dict()
        ubs = dict()
        vbs = dict()

        bounds_dictified = Boundary.to_dict(bounds).values() if dictify else bounds
        for b in bounds_dictified:

            if b.type == "inflow":      ptype, utype, vtype = "der", "val", "val"
            if b.type == "outflow":     ptype, utype, vtype = "val", "der", "der"
            if b.type == "noslip":      ptype, utype, vtype = "der", "val", "val"
            if b.type == "slip": 
                ptype = "der"
                utype = "val" if b.side in ("E", "W") else "der"
                vtype = "der" if b.side in ("E", "W") else "val"
            
            pbs[ptype + b.side] = LowBound.concat(pbs.get(ptype + b.side), LowBound(b.side, ptype, b.xs, b.ys, b.valp))
            ubs[utype + b.side] = LowBound.concat(ubs.get(utype + b.side), LowBound(b.side, utype, b.xs, b.ys, b.valu))
            vbs[vtype + b.side] = LowBound.concat(vbs.get(vtype + b.side), LowBound(b.side, vtype, b.xs, b.ys, b.valv))

        return (pbs, ubs, vbs)

Boundaries = Dict[str, Boundary]

# Expand functions take field and return a 2 bigger field padded with ghost cell values.
# Also sets the values when directly on the boundary

def _bc_fill_corners_pipe(out:np.ndarray):
    out[0,0] = out[1,0]
    out[0,-1] = out[1,-1]
    out[-1,0] = out[-2,0]
    out[-1,-1] = out[-2,-1]

def bc_expand_velx(f:np.ndarray, bcs:LowBounds) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f

    for type_side, bc in bcs.items():
        x, y, val = bc.xs+1, bc.ys+1, bc.value
        # directly on the boundary
        if type_side == "valW":
            out[x, y] = val
            out[x-1, y] = val
        elif type_side == "valE":
            out[x+1, y] = val
            out[x+2, y] = val
        # boundary between cells
        elif type_side == "valS": out[x, y-1] = 2*val - out[x, y]
        elif type_side == "valN": out[x, y+1] = 2*val - out[x, y]
        # directly on the boundary, so that central der matches
        elif type_side == "derW": out[x-1, y] = out[x+1, y] - 2*val*dx
        elif type_side == "derE": out[x+2, y] = out[x,   y] + 2*val*dx
        elif type_side == "derS": out[x, y-1] = out[x, y] - val*dy
        elif type_side == "derN": out[x, y+1] = out[x, y] + val*dy

    _bc_fill_corners_pipe(out)
    return out

def bc_expand_vely(f:np.ndarray, bcs:LowBounds) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f

    for type_side, bc in bcs.items():
        x, y, val = bc.xs+1, bc.ys+1, bc.value
        # boundary between cells
        if type_side == "valW":   out[x-1, y] = 2*val - out[x, y]
        elif type_side == "valE": out[x+1, y] = 2*val - out[x, y]
        # directly on the boundary
        elif type_side == "valS":
            out[x, y] = val
            out[x, y-1] = val
        elif type_side == "valN":
            out[x, y+1] = val
            out[x, y+2] = val
        elif type_side == "derW": out[x-1, y] = out[x, y] - val*dy
        elif type_side == "derE": out[x+1, y] = out[x, y] + val*dy
        # directly on the boundary, so that central der matches
        elif type_side == "derS": out[x, y-1] = out[x, y+1] - 2*val*dx
        elif type_side == "derN": out[x, y+2] = out[x, y  ] + 2*val*dx

    _bc_fill_corners_pipe(out)
    return out

def bc_expand_cell(f:np.ndarray, bcs:LowBounds) -> np.ndarray:
    out = np.zeros((f.shape[0] + 2, f.shape[1] + 2), dtype=f.dtype)
    out[1:-1, 1:-1] = f

    for type_side, bc in bcs.items():
        x, y, val = bc.xs+1, bc.ys+1, bc.value
        if   type_side == "valW": out[x-1, y] = 2*val - out[x, y]
        elif type_side == "valE": out[x+1, y] = 2*val - out[x, y]
        elif type_side == "valS": out[x, y-1] = 2*val - out[x, y]
        elif type_side == "valN": out[x, y+1] = 2*val - out[x, y]
        elif type_side == "derW": out[x-1, y] = out[x, y] - val*dx
        elif type_side == "derE": out[x+1, y] = out[x, y] + val*dx
        elif type_side == "derS": out[x, y-1] = out[x, y] - val*dy
        elif type_side == "derN": out[x, y+1] = out[x, y] + val*dy

    _bc_fill_corners_pipe(out)
    return out

def bc_expand_vels(f:np.ndarray, bcs:Tuple[LowBounds, LowBounds]) -> np.ndarray:
    return [bc_expand_velx(f[0], bcs[0]), bc_expand_vely(f[1], bcs[1])]

# Only sets the values directly on the boundary
def bc_apply_velx(f:np.ndarray, bcs:LowBounds, copy=True) -> np.ndarray:
    if copy: f = f.copy()
    if bc := bcs.get("valW"): 
        f[bc.xs, bc.ys] = bc.value
    if bc := bcs.get("valE"): 
        f[bc.xs+1, bc.ys] = bc.value
    return f

def bc_apply_vely(f:np.ndarray, bcs:LowBounds, copy=True) -> np.ndarray:
    if copy: f = f.copy()
    if bc := bcs.get("valS"): f[bc.xs, bc.ys] = bc.value
    if bc := bcs.get("valN"): f[bc.xs, bc.ys+1] = bc.value
    return f

def bc_apply_vels(f:np.ndarray, bcs:Tuple[LowBounds, LowBounds], copy=True) -> np.ndarray:
    u = bc_apply_velx(f[0], bcs[0], copy=copy)
    v = bc_apply_vely(f[1], bcs[1], copy=copy)
    return [u, v]

bc_expand_vel = (bc_expand_velx, bc_expand_vely)
bc_apply_vel = (bc_apply_velx, bc_apply_vely)

def matrix_free_solve(A:Callable[np.ndarray, [np.ndarray]], b:np.ndarray, expandOffset:bool=True, x0:np.ndarray|None = None, rtol:float = 1e-3, maxiter:int = 200) -> Tuple[np.ndarray, int]:
    offset = A(np.zeros_like(b))
    Aoff = lambda x: (A(x.reshape(b.shape)) - offset).flat
    Boff = (b - offset).flat

    Aop = sp.sparse.linalg.LinearOperator((len(Boff), len(Boff)), matvec=Aoff, dtype=b.dtype)
    xflat = x0.flat if x0 is not None else None
    xN, iters = sp.sparse.linalg.cgs(Aop, Boff, x0=xflat, rtol=rtol, atol=1e-3, maxiter=maxiter)
    if iters == 0:
        return xN.reshape(b.shape), iters
    else:
        return np.zeros_like(b), iters

def is_normal(x:np.ndarray) -> bool: return np.any(np.isinf(x) | np.isnan(x)) == False

# Difference operators taking expanded field and returning just the field (eliminates ghost cells)

# interpolate (for Interpolate_velx_to_vely: input x field, output interpolated to y field for Interpolate_vely_to_velx in reverse)
def Interpolate_velx_to_vely(f:np.ndarray) -> np.ndarray: 
    # base to match shapes: [1:-1, 1:-1] -> [1:-2,1:]
    # interp between [0, -1], [0, 0], [1, -1], [1, 0] thus the below ranges (add to base shape)
    return 1/4*(f[1:-2,:-1] + f[1:-2,1:] + f[2:-1,:-1] + f[2:-1, 1:])

def Interpolate_vely_to_velx(f:np.ndarray) -> np.ndarray: 
    # base to match shapes: [1:-1, 1:-1] -> [1:,1:-2]
    # interp between [-1, 0], [0, 0], [-1, 1], [0, 1] thus the below ranges (add to base shape)
    # should be just transposed Interpolate_velx_to_vely shapes!
    return 1/4*(f[:-1,1:-2] + f[1:,1:-2] + f[:-1,2:-1] + f[1:,2:-1])

# Interpolate velocity field to cell 
def Interpolate_vels_to_cellx(f:np.ndarray) -> np.ndarray: return 1/2*(f[1:,:] + f[:-1,:])
def Interpolate_vels_to_celly(f:np.ndarray) -> np.ndarray: return 1/2*(f[:,1:] + f[:,:-1])
# def Interpolate_vels_to_cellx(f:np.ndarray) -> np.ndarray: return f[:-1,:]
# def Interpolate_vels_to_celly(f:np.ndarray) -> np.ndarray: return f[:,:-1]

# first derivative
def Centralx(f:np.ndarray) -> np.ndarray: return (f[2:, 1:-1] - f[:-2, 1:-1])/(2*dx)
def Centraly(f:np.ndarray) -> np.ndarray: return (f[1:-1, 2:] - f[1:-1, :-2])/(2*dy)

# second derivative
def Central2x(f:np.ndarray) -> np.ndarray: return np.diff(f[:,1:-1], n=2, axis=0)/dx**2
def Central2y(f:np.ndarray) -> np.ndarray: return np.diff(f[1:-1,:], n=2, axis=1)/dy**2

# first derivative upwind
def Upwindx(f:np.ndarray, dir_f = None) -> np.ndarray: 
    dir_f = f[1:-1, 1:-1] if dir_f is None else dir_f
    dxn = (f[1:-1, 1:-1] - f[:-2, 1:-1])/dx
    dxp = (f[2:, 1:-1] - f[1:-1, 1:-1])/dx
    mask = (dir_f >= 0)
    return np.where(mask, dxn, dxp)

def Upwindy(f:np.ndarray, dir_f = None) -> np.ndarray: 
    dir_f = f[1:-1, 1:-1] if dir_f is None else dir_f
    dxn = (f[1:-1, 1:-1] - f[1:-1, :-2])/dy
    dxp = (f[1:-1, 2:] - f[1:-1, 1:-1])/dy
    mask = (dir_f >= 0)
    return np.where(mask, dxn, dxp)

def FluxLimitx(u:np.ndarray, un:np.ndarray) -> np.ndarray:
    # N-2 size
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (un[:-2,:] - un[1:-1,:])/(un[1:-1,:] - un[2:,:])
    r = np.nan_to_num(r, posinf=1e9, neginf=-1e9, nan=0)
    psi = (r + np.abs(r)) / (1 + np.abs(r))

    # N-3 size
    uin = u[:-3, :]
    ui = u[1:-2, :]
    uip = u[2:-1, :]
    uipp = u[3:, :]

    # N-1
    # ai = 0.5*(a[1:,:] + a[:-1,:])

    # N-3
    u_L = ui  + 0.5*psi[:-1,:]*(ui - uin)
    u_R = uip - 0.5*psi[1:,:]*(uipp - uip)
    u_up = np.where(un[1:-2,:] >= 0, u_L, u_R)
    F = u_up

    # N-4
    Ddx = (F[1:, :] - F[:-1, :]) / dx

    #N-2
    # Fill in the rest with simple upwind
    out = Upwindx(u, un[1:-1, 1:-1])
    out[1:-1,:] = Ddx[:,1:-1]
    return out 

def FluxLimity(u:np.ndarray, un:np.ndarray) -> np.ndarray:
    # N-2 size
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (un[:,:-2] - un[:,1:-1])/(un[:,1:-1] - un[:,2:])
    r = np.nan_to_num(r, posinf=1e9, neginf=-1e9, nan=0)
    psi = (r + np.abs(r)) / (1 + np.abs(r))

    # N-3 size
    uin = u[:,:-3]
    ui = u[:,1:-2]
    uip = u[:,2:-1]
    uipp = u[:,3:]

    # N-1
    # ai = 0.5*(a[:,1:] + a[:,:-1])
    
    # N-3
    u_L = ui  + 0.5*psi[:,:-1]*(ui - uin)
    u_R = uip - 0.5*psi[:,1:]*(uipp - uip)
    u_up = np.where(un[:,1:-2] >= 0, u_L, u_R)
    F = u_up

    # N-4
    Ddy = (F[:,1:] - F[:,:-1]) / dy

    #N-2
    # Fill in the rest with simple upwind
    out = Upwindy(u, un[1:-1, 1:-1])
    out[:,1:-1] = Ddy[1:-1,:]
    return out

Interpolate_vels_swap  = (Interpolate_velx_to_vely,  Interpolate_vely_to_velx)
Interpolate_vels_to_cell = (Interpolate_vels_to_cellx, Interpolate_vels_to_celly)
Central = (Centralx,  Centraly)
Central2 = (Central2x, Central2y)
Upwind = (Upwindx, Upwindy)
FluxLimit = (FluxLimitx, FluxLimity)

def Grad(ex:np.ndarray)-> np.ndarray:       return [Centralx(ex),  Centraly(ex)]
def Div(ex:List[np.ndarray]) -> np.ndarray: return Centralx(ex[0]) + Centraly(ex[1])
def Lap(ex:np.ndarray) -> np.ndarray:       return Central2x(ex) + Central2y(ex)

def Div_vels_to_cell(ex:List[np.ndarray]) -> np.ndarray:
    div = 0 
    div += np.diff(ex[0], n=1, axis=0)/dx
    div += np.diff(ex[1], n=1, axis=1)/dy 
    return div
    
def Grad_cell_to_vels(ex:np.ndarray) -> List[np.ndarray, np.ndarray]:
    grad = [None, None]
    grad[0] = np.diff(ex[:,1:-1], n=1, axis=0)/dx
    grad[1] = np.diff(ex[1:-1,:], n=1, axis=1)/dy 
    return grad

@dataclass
class AdvectParams:
    variant = "flux" #whether to do proper flux limiting. If true no other setting apply
    factor = 0.0 #0 to 1: only upwind to only central difference
    dynamic = 0.0 #0 to 1: no to only influence of (inproper) van-leer
    minblend = 0.0 
    maxblend = 1.0
    # def __init__(self): pass

# High level advect rutine routing/blending between the possible options (upwind, central, flux limmiting)
def Advect12(u:np.ndarray, un:np.ndarray, params:AdvectParams, d) -> np.ndarray:
    # proper flux limitting
    if params.variant == "flux":
        return FluxLimit[d](u, un)

    # Ad hoc flux limitting
    psi = 0
    if params.dynamic > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            if d == 0: r = (un[:-2,1:-1] - un[1:-1,1:-1])/(un[1:-1,1:-1] - un[2:,1:-1])
            if d == 1: r = (un[1:-1,:-2] - un[1:-1,1:-1])/(un[1:-1,1:-1] - un[1:-1,2:])
        
        r = np.nan_to_num(r, posinf=1e9, neginf=-1e9, nan=0)
        psi = (r + np.abs(r)) / (1 + np.abs(r))
        psi = np.minimum(psi, 1.0)

    upw = Upwind[d](u, un[1:-1, 1:-1])
    cen = Central[d](u)
    blend = params.factor + params.dynamic*(psi - params.factor) 
    blend = np.clip(blend, params.minblend, params.maxblend)
    return upw + blend*(cen - upw)
    
def Advectx(u:np.ndarray, un:np.ndarray, params:AdvectParams) -> np.ndarray: return Advect12(u, un, params, 0)
def Advecty(u:np.ndarray, un:np.ndarray, params:AdvectParams) -> np.ndarray: return Advect12(u, un, params, 1)

Advect = (Advectx, Advecty)

def step(un:np.ndarray, pn:np.ndarray, BCu:LowBounds, BCp:LowBounds, proj_variant:str, advect_norm:AdvectParams, advect_tang:AdvectParams, rtol=1e-3, proj_iters=1):
    # Exact:      du/dt = - dot(u, div(u)) + nu*lap(u) - div(p)/rho + S
    # Discrete t: 
    # (un - u)/dt = - dot(u, div(un)) + nu*lap(un) - div(p)/rho + S
    # un - u = - dt*dot(u, div(un)) + dt*nu*lap(un) - dt*div(p)/rho + dt*S
    # un + dt*dot(u, div(un)) - dt*nu*lap(un) = u - dt*div(p)/rho + dt*S
    
    # un_kn = A^-1(f_rhs - G*pn_k)
    # qn_kn = rho/dt*L^-1*Central*un_kn
    # pn_kn = pn_k + qn_kn - mu/rho*Central*un_kn
    un = bc_apply_vels(un, BCu)

    unk = un
    pnk = pn #todo guess
    for k in range(proj_iters):
        unknorm = bc_expand_vels(unk, BCu)
        #interpolated one field onto the other and expanded
        # according to the others boundary conditons.
        unktang = [ 
            bc_expand_vely(Interpolate_velx_to_vely(unknorm[0]), BCu[1]),
            bc_expand_velx(Interpolate_vely_to_velx(unknorm[1]), BCu[0]),
        ]

        S = [0, 0] #source terms
        grad_p = [0, 0]
        if proj_variant != "non-increment":
            pex = bc_expand_cell(pnk, BCp)
            grad_p = Grad_cell_to_vels(pex)
            
        u_pred = [0, 0]
        for d in range(2):
            t = 1-d
            def pred_lhs(u:np.ndarray) -> np.ndarray:
                uex = bc_expand_vel[d](u, BCu[d])
                advnorm = unknorm[d][1:-1, 1:-1]*Advect[d](uex, unknorm[d], advect_norm)
                advtang = unktang[t][1:-1, 1:-1]*Advect[t](uex, unktang[t], advect_tang)
                adv = advnorm + advtang

                dif = nu*Lap(uex)
                U = u + dt*adv - dt*dif
                U = bc_apply_vel[d](U, BCu[d], copy=False)
                return U

            pred_rhs = un[d] - dt*grad_p[d]/rho + dt*S[d]
            u_pred[d], predIters = matrix_free_solve(pred_lhs, pred_rhs, x0=un[d], rtol=rtol)
            if predIters != 0:
                print("Predictor diverged!")
                return (unk, pnk, np.zeros_like(pnk), np.zeros_like(pnk), unk)
            
        u_pred = bc_apply_vels(u_pred, BCu, copy=False) 
        div_u_pred = Div_vels_to_cell(u_pred)

        corr_lhs = lambda p: Lap(bc_expand_cell(p, BCp))
        corr_rhs = 1/dt*div_u_pred
        p_corr, corrIters = matrix_free_solve(corr_lhs, corr_rhs)
        if corrIters != 0:
            print("Corrector diverged!")
            return (unk, pnk, div_u_pred, p_corr, u_pred)
        
        p_correx = bc_expand_cell(p_corr, BCp)
        grad_p_corr = Grad_cell_to_vels(p_correx)

        u_next = [None, None]
        u_next[0] = u_pred[0] - dt*grad_p_corr[0]
        u_next[1] = u_pred[1] - dt*grad_p_corr[1]
        u_next = bc_apply_vels(u_next, BCu, copy=False)

        if   proj_variant == "non-increment":    p_next = p_corr
        elif proj_variant == "increment":        p_next = pnk + p_corr
        elif proj_variant == "increment-rot":    p_next = pnk + p_corr - nu*div_u_pred

        unk = u_next
        pnk = p_next

    assert un[0].shape == unk[0].shape
    assert un[1].shape == unk[1].shape
    assert pnk.shape == pnk.shape
    return (unk, pnk, div_u_pred, p_corr, u_pred)

def main():
    # PARAM SETTING
    global nx, ny, nu, Lx, Ly, dx, dy, dt
    nx = 40 #num cells
    ny = 40
    nu = 1.3059e-5 #viscosity
    Ly = 1 #size of domain in meters
    Lx = Ly*nx/ny 
    dx = Lx/nx
    dy = Ly/ny
    dt = 4e-3
    t0 = 0
    t1 = 100
    rtol = 1e-3
    proj_iters = 1 
    show_interval = 0
    plot_every = 50

    # domain = "channel"
    # domain = "cavity"
    domain = "channel_cavity"
    # domain = "real_cavity"

    proj_variant = "non-increment"
    # proj_variant = "increment"
    # proj_variant = "increment-rot"

    advect_norm = AdvectParams() 
    # advect_norm.variant = "flux"
    advect_norm.variant = "blend"
    advect_norm.factor = 0.97
    advect_norm.dynamic = 0.0
    advect_norm.minblend = 0.0 
    advect_norm.maxblend = 1.0
    
    advect_tang = AdvectParams() 
    # advect_tang.variant = "flux"
    advect_tang.variant = "blend"
    advect_tang.factor = 0.97
    advect_tang.dynamic = 0.0
    advect_tang.minblend = 0.0 
    advect_tang.maxblend = 1.0

    example_fields = False
    # example_fields = True

    # display_field = ""
    # display_field = "p"
    # display_field = "u"
    # display_field = "v"
    display_field = "velmag"
    # display_field = "predu"
    # display_field = "predv"
    # display_field = "predmag"
    # display_field = "divpred"
    # display_field = "corrpred"
    # display_field = "lapcorrpred"
    # display_field = "psiu"
    # display_field = "psiv"
    # display_field = "psimag"

    display_cell_centers = False
    display_grid = False
    display_BCs = True
    display_velocity_arrows = False
    display_face_velocity_arrows = False
    display_streamlines = True 

    dd = min(dx, dy)
    ARROW_SCALE = 1/dd

    # initial conditions
    @dataclass
    class Fields:
        p:np.ndarray
        u:np.ndarray
        v:np.ndarray

    fields = Fields(
        p = np.zeros((nx, ny)),
        u = np.zeros((nx+1, ny)),
        v = np.zeros((nx, ny+1))
    )

    #grid params
    x_centers = (np.arange(nx) + 0.5) * dx
    y_centers = (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')

    xfu = np.arange(nx + 1) * dx
    yfu = (np.arange(ny) + 0.5) * dy
    Xu, Yu = np.meshgrid(xfu, yfu, indexing='ij')

    xfv = (np.arange(nx) + 0.5) * dx
    yfv = np.arange(ny + 1) * dy
    Xv, Yv = np.meshgrid(xfv, yfv, indexing='ij')

    xfuex = (np.arange(nx+3) - 1) * dx
    yfuex = (np.arange(ny+2) - 1 + 0.5) * dy
    Xuex, Yuex = np.meshgrid(xfuex, yfuex, indexing='ij')

    xfvex = (np.arange(nx+2) - 1 + 0.5) * dx
    yfvex = (np.arange(ny+3) - 1) * dy
    Xvex, Yvex = np.meshgrid(xfvex, yfvex, indexing='ij')

    # Main loop
    fig = None
    iter = -1
    t = t0
    while True:
        iter += 1
        t = iter*dt
        if t > t1:
            break
        
        def parabolic_profile(u:float, n:int) -> np.ndarray:
            centers = (np.arange(n) + 0.5)/n
            profile = u*(1 - (2*centers - 1)**2)
            return profile

        # BOUNDARIES ===============
        u_in = min(1, 10*t)
        if domain == "channel":
            inflow = u_in
            inflow = parabolic_profile(u_in, ny)

            boundaries = Boundary.to_dict([
                Boundary.inflow("W", 0, np.arange(ny), inflow, 0),
                Boundary.outflow("E", nx-1, np.arange(ny)),
                Boundary.noslip("S", np.arange(nx), 0),
                Boundary.noslip("N", np.arange(nx), ny-1),
            ])

        elif domain == "cavity":
            boundaries = Boundary.to_dict([
                Boundary.noslip("W", 0, np.arange(ny)),
                Boundary.noslip("E", nx-1, np.arange(ny)),
                Boundary.noslip("S", np.arange(nx), 0),
                Boundary.inflow("N", np.arange(nx), ny-1, u_in, 0),
            ])
        elif domain == "real_cavity":
            gapW = 3
            gapE = 2
            # inflow = parabolic_profile(u_in, gapW)
            inflow = u_in
            boundaries = Boundary.to_dict([
                Boundary.inflow("W", 0, np.arange(ny-gapW, ny), inflow, 0),
                Boundary.noslip("W", 0, np.arange(0, ny-gapW)),
                
                Boundary.outflow("E", nx-1, np.arange(ny-gapE, ny)),
                Boundary.slip("E", nx-1, np.arange(0, ny-gapE)),

                Boundary.slip("N", np.arange(nx), ny-1),
                Boundary.noslip("S", np.arange(nx), 0),
            ])

        elif domain == "channel_cavity":
            inflow = u_in
            gap = max(ny//5, 1)
            inflow = parabolic_profile(u_in, gap)

            boundaries = Boundary.to_dict([
                Boundary.inflow("W", 0, np.arange(ny-gap, ny), inflow, 0),
                Boundary.noslip("W", 0, np.arange(0, ny-gap)),
                Boundary.slip("E", nx-1, np.arange(gap, ny)),
                Boundary.outflow("E", nx-1, np.arange(0, gap)),
                Boundary.noslip("S", np.arange(nx), 0),
                Boundary.noslip("N", np.arange(nx), ny-1),
            ])
        else:
            assert False

        BCp, BCu, BCv = Boundary.to_low_bounds(boundaries.values(), dictify=False)
        BCp, BCu, BCv = Boundary.to_low_bounds(boundaries.values(), dictify=False)

        step_out = [None, None, None, None]
        # SIMULATE ============================
        if example_fields:
            fields.u = 0.6*np.sin(np.pi * Yu/(ny * dy)) + 0.2*(0.5 - Xu/(nx * dx))
            fields.v = 0.6*np.cos(np.pi * Xv/(nx * dx)) + 0.2*(0.5 - Yv/(ny * dy))
            fields.p = np.sin(np.pi * Xc/(nx * dx)) * np.cos(np.pi * Yc/(ny * dy))
        else:
            # pass
            step_out = step([fields.u, fields.v], fields.p, [BCu, BCv], BCp, 
                proj_variant=proj_variant, 
                proj_iters=proj_iters, 
                advect_norm=advect_norm, 
                advect_tang=advect_tang)
            fields.u = step_out[0][0]
            fields.v = step_out[0][1]
            fields.p = step_out[1]

        #PLOTTING ============================
        if iter == 1 or iter % plot_every == 0:
            pex = bc_expand_cell(fields.p, BCp)
            uex = bc_expand_velx(fields.u, BCu)
            vex = bc_expand_vely(fields.v, BCv)

            if fig is None:
                plt.ion()
                fig = plt.figure(figsize=(12, 10), dpi=100)

            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_xlim(-dx, (nx + 1) * dx)
            ax.set_ylim(-dy, (ny + 1) * dy)
            ax.set_aspect('equal')
            ax.set_title(f"iter = {iter} t = {float(t):.6} cfl = {get_cfl(fields.u)}")

            velx = Interpolate_vels_to_cellx(uex)
            vely = Interpolate_vels_to_celly(vex)
            # Field drawing
            display_field_tuple = None
            if   display_field == "p":          display_field_tuple = (pex, "pressure")
            elif display_field == "u":      display_field_tuple = (velx, "velocity u")
            elif display_field == "v":      display_field_tuple = (vely, "velocity v")
            elif display_field == "velmag": display_field_tuple = (np.hypot(velx, vely), "velocity magnitude")
            elif display_field == "predu":
                display_field_tuple = (Interpolate_vels_to_cellx(bc_expand_velx(step_out[4][0], BCu)), "predictor u")
            elif display_field == "predv":
                display_field_tuple = (Interpolate_vels_to_celly(bc_expand_vely(step_out[4][1], BCv)), "predictor v")
            elif display_field == "predmag":
                epred = bc_expand_vels(step_out[4], (BCu, BCv))
                iepred = [Interpolate_vels_to_cellx(epred[0]), Interpolate_vels_to_celly(epred[1])]
                iepredmag = np.hypot(iepred[0], iepred[1])
                display_field_tuple = (iepredmag, "predictor magnitude")
            elif display_field == "divpred":
                display_field_tuple = (bc_expand_cell(step_out[2], {}), "divpred")
            elif display_field == "corrpred":
                display_field_tuple = (bc_expand_cell(step_out[3], BCp), "corrpred")
            elif display_field == "lapcorrpred":
                corrpred_lap = bc_expand_cell(Lap(bc_expand_cell(step_out[3], BCp)), {})
                display_field_tuple = (corrpred_lap, "corrpred lap")
            elif display_field in ["psiu", "psiv", "psimag"]:
                ru = (uex[:-2,1:-1] - uex[1:-1,1:-1])/(uex[1:-1,1:-1] - uex[2:,1:-1])
                rv = (vex[1:-1,:-2] - vex[1:-1,1:-1])/(vex[1:-1,1:-1] - vex[1:-1,2:])

                psiu = np.zeros_like(uex)
                psiu[1:-1, 1:-1] = np.minimum((ru + np.abs(ru)) / (1 + np.abs(ru)), 1)
                
                psiv = np.zeros_like(vex)
                psiv[1:-1, 1:-1] = np.minimum((rv + np.abs(rv)) / (1 + np.abs(rv)), 1)
                
                ipsi = [Interpolate_vels_to_cellx(psiu), Interpolate_vels_to_celly(psiv)]
                if display_field == "psiu": display_field_tuple = (psiu, "psiu")
                if display_field == "psiv": display_field_tuple = (psiv, "psiv")
                if display_field == "psimag": display_field_tuple = (np.hypot(ipsi[0], ipsi[1]), "psimag")

            if display_field_tuple is not None:
                im = ax.imshow(display_field_tuple[0].T, origin='lower', extent=[-dx, (nx+1)*dx, -dy, (ny+1)*dy], interpolation='nearest')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(display_field_tuple[1])

            # markers for cell centers and faces
            if display_cell_centers:
                ax.scatter(Xc.flatten(), Yc.flatten(), marker='o', color='red', s=30)
            if display_grid:
                ax.plot([xfu, xfu], [np.full(nx+1, 0), np.full(nx+1, Ly)], color='black', linewidth=0.4)
                ax.plot([np.full(ny+1, Lx), np.full(ny+1, 0)], [yfv, yfv], color='black', linewidth=0.4)
            if display_streamlines:
                ax.streamplot(Xc[:,0], Yc[0,:], velx[1:-1,1:-1].T, vely[1:-1,1:-1].T, color="black", density=1, linewidth=0.8, arrowsize=0.7)
            if display_velocity_arrows:
                ax.quiver(Xc, Yc, velx[1:-1,1:-1], vely[1:-1,1:-1], angles='xy', scale_units='xy', scale=ARROW_SCALE, width=dd*0.02, headwidth=5)

            # velocity arrows
            if display_face_velocity_arrows:
                YuexStaggered = Yuex.copy()
                YuexStaggered[1::2,:] -= 0.05*dy
                YuexStaggered[0::2,:] += 0.05*dy

                XvexStaggered = Xvex.copy()
                XvexStaggered[:,1::2] -= 0.05*dx
                XvexStaggered[:,0::2] += 0.05*dx
                ax.quiver(Xuex, YuexStaggered, uex, np.zeros_like(uex), angles='xy', scale_units='xy', scale=ARROW_SCALE, width=dd*0.02, headwidth=5)
                ax.quiver(XvexStaggered, Yvex, np.zeros_like(vex), vex, angles='xy', scale_units='xy', scale=ARROW_SCALE, width=dd*0.02, headwidth=5)

            # boundaries
            for b in boundaries.values():
                if display_BCs == False:
                    continue

                # TODO: get rid of this!
                @dataclass
                class BoundRenderInfo: 
                    dim:str
                    n:list #normal, tangential direction
                    t:list 
                    ddn:float  #dx or dy in normal or tan dir
                    ddt:float 
                    xs:np.ndarray #bottom left point of boundary face x,y
                    ys:np.ndarray
                    
                if b.side == "W": info = BoundRenderInfo("x", [-1, 0], [0, 1], dx, dy, b.xs, b.ys)
                if b.side == "E": info = BoundRenderInfo("x", [1, 0], [0, -1], dx, dy, b.xs+1, b.ys)
                if b.side == "S": info = BoundRenderInfo("y", [0, -1], [1, 0], dy, dx, b.xs, b.ys)
                if b.side == "N": info = BoundRenderInfo("y", [0, 1], [-1, 0], dy, dx, b.xs, b.ys+1)

                n, t = np.array(info.n), np.array(info.t)
                xs, ys = info.xs, info.ys
                ddn, ddt = info.ddn, info.ddt
                vx, vy = b.valu, b.valv
                v = np.vstack((vx, vy)).T 

                coords = np.array([xs, ys]).T
                dx_dy = np.array([dx, dy])
                # face edges, face center
                e1 = coords * dx_dy
                e2 = (coords + np.abs(info.t)) * dx_dy
                ec = (e1 + e2) / 2

                if b.type == "noslip":
                    xs = np.array([e1[:, 0], e2[:, 0]]).T
                    ys = np.array([e1[:, 1], e2[:, 1]]).T
                    ax.plot(xs, ys, color="black", solid_capstyle='butt', linewidth=4)

                if b.type == "slip":
                    xs = np.array([e1[:, 0], e2[:, 0]]).T
                    ys = np.array([e1[:, 1], e2[:, 1]]).T
                    ax.plot(xs, ys, color="black", solid_capstyle='butt', linewidth=4, linestyle=":")

                if b.type == "inflow":
                    if np.all(v*n == 0):
                        soffx, soffy = ec[:,0], ec[:,1]
                        if info.dim == "y": 
                            soffy[0::2] += 0.1*ddn
                            soffy[1::2] += 0.05*ddn
                        if info.dim == "x": 
                            soffx[0::2] += 0.1*ddn
                            soffx[1::2] += 0.05*ddn
                        ax.quiver(soffx, soffy, vx, vy, angles='xy', scale_units='xy', scale=ARROW_SCALE, width=dd*0.02, headwidth=5)

                        xs = np.array([e1[:, 0], e2[:, 0]]).T
                        ys = np.array([e1[:, 1], e2[:, 1]]).T
                        ax.plot(xs, ys, color="black", solid_capstyle='butt', linestyle=':', linewidth=1.5)
                    else:
                        for off in np.linspace(-0.4, 0.4, 5)*ddt:
                            o = ec-v*dd + off*t
                            ax.quiver(o[:,0], o[:,1], vx, vy, angles='xy', scale_units='xy', scale=ARROW_SCALE, width=dd*0.02, headwidth=5)

                if b.type == "outflow":
                    # offsets = [0, 0.1, 0.2]
                    # styles = ["solid", "solid", "dotted"]
                    
                    offsets = [0, 0.2]
                    styles = ["-", (0, (2, 3))]
                    for style, off in zip(styles, offsets):
                        o1 = e1 + off*n*ddn
                        o2 = e2 + off*n*ddn

                        xs = np.array([o1[:, 0], o2[:, 0]]).T
                        ys = np.array([o1[:, 1], o2[:, 1]]).T
                        ax.plot(xs, ys, color="black", solid_capstyle='butt', linestyle=style, linewidth=1)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(show_interval) 

    plt.ioff()
    plt.show()

main()
