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
beta = 0.002
# phi_eps = 0.1
phi_width = 8
phi_eps = 0.001
phi_delta = 1e-4
phi_cutoff = 0.4

def get_cfl(u):
    return np.max(np.abs(u)*dt/dx)

def get_mach_number(u):
    return np.max(np.abs(u)/c_sound)

Side         = Literal["N", "S", "E", "W"]
LowBoundType = Literal["val", "der"]
BoundaryType = Literal["inflow", "outflow", "wall"]

Cell = np.ndarray
Velx = np.ndarray
Vely = np.ndarray
Vels = List[np.ndarray]
CellEx = np.ndarray
VelxEx = np.ndarray
VelyEx = np.ndarray
VelsEx = List[np.ndarray]

Field = Union[Cell, Velx, Vels]
FieldEx = Union[CellEx, VelxEx, VelsEx]

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
        ubs = {}
        vbs = {}
        pbs = {}
        fbs = {}

        bounds_dictified = Boundary.to_dict(bounds).values() if dictify else bounds
        for b in bounds_dictified:

            if b.type == "inflow":      utype, vtype, ptype, ftype = "val", "val", "der", "der"
            if b.type == "outflow":     utype, vtype, ptype, ftype = "der", "der", "val", "der"
            if b.type == "noslip":      utype, vtype, ptype, ftype = "val", "val", "der", "der"
            if b.type == "slip": 
                ptype, ftype = "der", "der"
                utype = "val" if b.side in ("E", "W") else "der"
                vtype = "der" if b.side in ("E", "W") else "val"
            
            pbs[ptype + b.side] = LowBound.concat(pbs.get(ptype + b.side), LowBound(b.side, ptype, b.xs, b.ys, b.valp))
            ubs[utype + b.side] = LowBound.concat(ubs.get(utype + b.side), LowBound(b.side, utype, b.xs, b.ys, b.valu))
            vbs[vtype + b.side] = LowBound.concat(vbs.get(vtype + b.side), LowBound(b.side, vtype, b.xs, b.ys, b.valv))
            fbs[ftype + b.side] = LowBound.concat(fbs.get(ftype + b.side), LowBound(b.side, ftype, b.xs, b.ys, np.zeros_like(b.ys)))
        return {
            'u':ubs, 
            'v':vbs, 
            'p':pbs, 
            'f':fbs
        }

Boundaries = Dict[str, Boundary]

# Expand functions take field and return a 2 bigger field padded with ghost cell values.
# Also sets the values when directly on the boundary

def _bc_fill_corners_pipe(out:np.ndarray):
    out[0,0] = out[1,0]
    out[0,-1] = out[1,-1]
    out[-1,0] = out[-2,0]
    out[-1,-1] = out[-2,-1]

def bc_expand_velx(f:Velx, bcs:LowBounds) -> VelxEx:
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

def bc_expand_vely(f:Vely, bcs:LowBounds) -> VelyEx:
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

def bc_expand_cell(f:Cell, bcs:LowBounds, out=None) -> CellEx:
    if out is None:
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

def bc_expand_vels(f:Vels, bcs:Tuple[LowBounds, LowBounds]) -> VelsEx:
    return [bc_expand_velx(f[0], bcs[0]), bc_expand_vely(f[1], bcs[1])]

# Only sets the values directly on the boundary
def bc_apply_velx(f:Velx, bcs:LowBounds, copy=True) -> Velx:
    if copy: f = f.copy()
    if bc := bcs.get("valW"): 
        f[bc.xs, bc.ys] = bc.value
    if bc := bcs.get("valE"): 
        f[bc.xs+1, bc.ys] = bc.value
    return f

def bc_apply_vely(f:Vely, bcs:LowBounds, copy=True) -> Vely:
    if copy: f = f.copy()
    if bc := bcs.get("valS"): f[bc.xs, bc.ys] = bc.value
    if bc := bcs.get("valN"): f[bc.xs, bc.ys+1] = bc.value
    return f

def bc_apply_vels(f:Vels, bcs:Tuple[LowBounds, LowBounds], copy=True) -> Vels:
    u = bc_apply_velx(f[0], bcs[0], copy=copy)
    v = bc_apply_vely(f[1], bcs[1], copy=copy)
    return [u, v]

bc_expand_vel = (bc_expand_velx, bc_expand_vely)
bc_apply_vel = (bc_apply_velx, bc_apply_vely)

def matrix_free_solve(A:Callable[np.ndarray, [np.ndarray]], b:np.ndarray, x0:np.ndarray|None = None, rtol:float = 1e-5, atol:float=0, maxiter:int = 300) -> Tuple[np.ndarray, int]:
    offset = A(np.zeros_like(b))
    Aoff = lambda x: (A(x.reshape(b.shape)) - offset).flat
    Boff = (b - offset).flat

    Aop = sp.sparse.linalg.LinearOperator((len(Boff), len(Boff)), matvec=Aoff, dtype=b.dtype)
    xflat = x0.flat if x0 is not None else None
    xN, iters = sp.sparse.linalg.bicgstab(Aop, Boff, x0=xflat, rtol=rtol, atol=atol, maxiter=maxiter)
    # xN, iters = sp.sparse.linalg.cgs(Aop, Boff, x0=xflat, rtol=rtol, atol=atol, maxiter=maxiter)

    # iters<0: breakdown (solution unreliable)
    if iters < 0:
        return np.zeros_like(b), iters
    # iters==0: converged; iters>0: maxiter hit (partial solution still useful);
    return xN.reshape(b.shape), iters

# def is_normal(x:np.ndarray) -> bool: return np.any(np.isinf(x) | np.isnan(x)) == False

# Difference operators taking expanded field and returning just the field (eliminates ghost cells)

# interpolate (for Interp_velx_to_vely: input x field, output interpolated to y field for Interp_vely_to_velx in reverse)
def Interp_velx_to_vely(f:VelxEx) -> Vely: 
    # base to match shapes: [1:-1, 1:-1] -> [1:-2,1:]
    # interp between [0, -1], [0, 0], [1, -1], [1, 0] thus the below ranges (add to base shape)
    return 1/4*(f[1:-2,:-1] + f[1:-2,1:] + f[2:-1,:-1] + f[2:-1, 1:])

def Interp_vely_to_velx(f:VelyEx) -> Velx: 
    # base to match shapes: [1:-1, 1:-1] -> [1:,1:-2]
    # interp between [-1, 0], [0, 0], [-1, 1], [0, 1] thus the below ranges (add to base shape)
    # should be just transposed Interp_velx_to_vely shapes!
    return 1/4*(f[:-1,1:-2] + f[1:,1:-2] + f[:-1,2:-1] + f[1:,2:-1])

# Interpolate velocity field to cell 
def Interp_vels_to_cellx(f:VelxEx) -> Cell: return 1/2*(f[1:,:] + f[:-1,:])
def Interp_vels_to_celly(f:VelyEx) -> Cell: return 1/2*(f[:,1:] + f[:,:-1])
def Interp_vels_to_cell_nearestx(f:VelxEx) -> Cell: return f[:-1,:]
def Interp_vels_to_cell_nearesty(f:VelyEx) -> Cell: return f[:,:-1]
def Interp_cell_to_velx(c:CellEx) -> Velx: return 0.5*(c[:-1,1:-1] + c[1:,1:-1])
def Interp_cell_to_vely(c:CellEx) -> Vely: return 0.5*(c[1:-1,:-1] + c[1:-1,1:])
def Interp_cell_to_velx_e(c:CellEx) -> Velx: return 0.5*(c[:-1,:] + c[1:,:])
def Interp_cell_to_vely_e(c:CellEx) -> Vely: return 0.5*(c[:,:-1] + c[:,1:])

def Interp_cell_to_vels(c:CellEx) -> List[np.ndarray]: 
    return [Interp_cell_to_velx(c), Interp_cell_to_vely(c)]

# first derivative
def Centralx(f:FieldEx) -> Field: return (f[2:, 1:-1] - f[:-2, 1:-1])/(2*dx)
def Centraly(f:FieldEx) -> Field: return (f[1:-1, 2:] - f[1:-1, :-2])/(2*dy)

# second derivative
def Central2x(f:FieldEx) -> Field: return np.diff(f[:,1:-1], n=2, axis=0)/dx**2
def Central2y(f:FieldEx) -> Field: return np.diff(f[1:-1,:], n=2, axis=1)/dy**2

def DivGradfx(faces: np.ndarray, u: np.ndarray) -> np.ndarray:
    du = np.diff(u[:, 1:-1], axis=0)   # u[1:,1:-1] - u[:-1,1:-1]
    np.multiply(du, faces, out=du)     # du *= faces
    return np.diff(du, axis=0) / dx**2

def DivGradfy(faces: np.ndarray, u: np.ndarray) -> np.ndarray:
    du = np.diff(u[1:-1, :], axis=1)   # u[1:-1,1:] - u[1:-1,:-1]
    np.multiply(du, faces, out=du)     # du *= faces
    return np.diff(du, axis=1) / dy**2

# def DivGrad(faces:List[np.ndarray], cells:FieldEx) -> Field: 
    # return DivGradfx(faces[0], cells) + DivGradfy(faces[1], cells) 

# Optimized version 
def DivGrad(faces, cells, out=None, scratch_x=None, scratch_y=None):
    fx, fy = faces
    dtype = np.result_type(cells, fx, fy, 0.0)
    if out is None:
        out = np.empty((cells.shape[0] - 2, cells.shape[1] - 2), dtype=dtype)
    if scratch_x is None:
        scratch_x = np.empty_like(fx, dtype=dtype)
    if scratch_y is None:
        scratch_y = np.empty_like(fy, dtype=dtype)

    np.subtract(cells[1:, 1:-1], cells[:-1, 1:-1], out=scratch_x)
    np.multiply(scratch_x, fx, out=scratch_x)
    np.subtract(scratch_x[1:], scratch_x[:-1], out=out)
    out *= 1.0 / dx**2

    np.subtract(cells[1:-1, 1:], cells[1:-1, :-1], out=scratch_y)
    np.multiply(scratch_y, fy, out=scratch_y)
    np.subtract(scratch_y[:, 1:], scratch_y[:, :-1], out=scratch_x[:-1])
    scratch_x[:-1] *= 1.0 / dy**2
    out += scratch_x[:-1]

    return out
    
# first derivative upwind
def Upwindx(f:VelxEx, dir_f:VelxEx = None) -> Velx: 
    dir_f = f[1:-1, 1:-1] if dir_f is None else dir_f
    dxn = (f[1:-1, 1:-1] - f[:-2, 1:-1])/dx
    dxp = (f[2:, 1:-1] - f[1:-1, 1:-1])/dx
    mask = (dir_f >= 0)
    return np.where(mask, dxn, dxp)

def Upwindy(f:VelyEx, dir_f:VelyEx = None) -> Vely: 
    dir_f = f[1:-1, 1:-1] if dir_f is None else dir_f
    dxn = (f[1:-1, 1:-1] - f[1:-1, :-2])/dy
    dxp = (f[1:-1, 2:] - f[1:-1, 1:-1])/dy
    mask = (dir_f >= 0)
    return np.where(mask, dxn, dxp)

def FluxLimitx(u:VelxEx, un:VelxEx) -> Velx:
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

def FluxLimity(u:VelyEx, un:VelyEx) -> Vely:
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

Interp_vels_swap  = (Interp_velx_to_vely,  Interp_vely_to_velx)
Interp_vels_to_cell = (Interp_vels_to_cellx, Interp_vels_to_celly)
Central = (Centralx,  Centraly)
Central2 = (Central2x, Central2y)
Upwind = (Upwindx, Upwindy)
FluxLimit = (FluxLimitx, FluxLimity)



def Grad(ex:FieldEx)-> Field:       return [Centralx(ex),  Centraly(ex)]
def Div(ex:List[FieldEx]) -> Field: return Centralx(ex[0]) + Centraly(ex[1])
def Lap(ex:FieldEx) -> Field:       return Central2x(ex) + Central2y(ex)

def Div_vels_to_cell(ex:Vels) -> Cell:
    div = 0 
    div += np.diff(ex[0], n=1, axis=0)/dx
    div += np.diff(ex[1], n=1, axis=1)/dy 
    return div
    
def Grad_cell_to_vels(ex:CellEx) -> Vels:
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

# High level advect rutine routing/blending between the possible options (upwind, central, flux limmiting)
def Advect12(u:np.ndarray, un:np.ndarray, params:AdvectParams, d) -> np.ndarray:
    # proper flux limitting
    if params.variant == "flux":
        return FluxLimit[d](u, un)

    # Ad hoc "flux limitting" via blending of upwind and central
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

def step(
    fields:dict, low_bounds:dict,
    advect_norm:AdvectParams, advect_tang:AdvectParams, 
    step = 0,
    proj_variant="increment-rot",
    proj_nonlinear=False,
    proj_iters=1) -> dict:
    
    BCu = [low_bounds['u'], low_bounds['v']]
    BCp = low_bounds['p']

    uxn:Velx = fields['u'] 
    uyn:Vely = fields['v'] 
    pn:Cell  = fields['p'] 
    
    un = bc_apply_vels([uxn, uyn], BCu)

    # Exact:      du/dt = - dot(u, div(u)) + nu*lap(u) - grad(p)/rho + S
    # Discrete t: 
    # (un - u)/dt = - dot(u, div(un)) + nu*lap(un) - grad(p)/rho + S
    # un - u = - dt*dot(u, div(un)) + dt*nu*lap(un) - dt*grad(p)/rho + dt*S
    # un + dt*dot(u, div(un)) - dt*nu*lap(un) = u - dt*grad(p)/rho + dt*S
    
    # un_kn = A^-1(f_rhs - G*pn_k)
    # qn_kn = rho/dt*L^-1*Central*un_kn
    # pn_kn = pn_k + qn_kn - mu/rho*Central*un_kn

    unk = un
    pnk = pn #todo guess

    # It doesnt make sense to do iterations on non-incremental scheme
    if proj_variant == "non-increment":
        proj_iters = 1

    for k in range(proj_iters):
        ex_unknorm = bc_expand_vels(unk if proj_nonlinear else un, BCu)
        #interpolated one field onto the other and expanded
        # according to the others boundary conditons.
        ex_unktang = [ 
            bc_expand_vely(Interp_velx_to_vely(ex_unknorm[0]), BCu[1]),
            bc_expand_velx(Interp_vely_to_velx(ex_unknorm[1]), BCu[0]),
        ]
        
        S = [0, 0] #source terms
        grad_pnk = [0, 0]
        if proj_variant != "non-increment":
            ex_p = bc_expand_cell(pnk, BCp)
            grad_pnk = Grad_cell_to_vels(ex_p)
            
        u_pred = [un[0], un[1]]
        for d in range(2):
            t = 1-d
            def pred_lhs(u:np.ndarray) -> np.ndarray:
                uex = bc_expand_vel[d](u, BCu[d])
                advnorm = ex_unknorm[d][1:-1, 1:-1]*Advect[d](uex, ex_unknorm[d], advect_norm)
                advtang = ex_unktang[t][1:-1, 1:-1]*Advect[t](uex, ex_unktang[t], advect_tang)
                adv = advnorm + advtang

                dif = nu*Lap(uex)
                U = u + dt*adv - dt*dif
                U = bc_apply_vel[d](U, BCu[d], copy=False)
                return U

            pred_rhs = un[d] - dt/rho*grad_pnk[d] + dt*S[d]
            pred_maxiter = max(300, 3 * pred_rhs.size)
            u_pred[d], pred_iters = matrix_free_solve(pred_lhs, pred_rhs, x0=unk[d], maxiter=pred_maxiter)
            if pred_iters < 0:
                print(f"Predictor ({d=}) breakdown at step {step}")
                return {"u": unk[0], "v": unk[1], "p":pnk, "u_pred": u_pred}
            if pred_iters > 0:
                print(f"Predictor ({d=}) slow convergence ({pred_iters} iters) at step {step}")
            
        u_pred = bc_apply_vels(u_pred, BCu, copy=False) 
        div_u_pred = Div_vels_to_cell(u_pred)

        if   proj_variant == "non-increment":    corr_guess = pnk
        elif proj_variant == "increment":        corr_guess = None
        elif proj_variant == "increment-rot":    corr_guess = nu*div_u_pred
        corr_lhs = lambda p: Lap(bc_expand_cell(p, BCp))
        corr_rhs = rho/dt*div_u_pred
        corr_maxiter = max(300, 3 * corr_rhs.size)
        p_corr, corr_iters = matrix_free_solve(corr_lhs, corr_rhs, x0=corr_guess, maxiter=corr_maxiter)
        if corr_iters < 0:
            print(f"Corrector breakdown at step {step}")
            return {"u": unk[0], "v": unk[1], "p":pnk, "u_pred": u_pred}
        if corr_iters > 0:
            print(f"Corrector slow convergence ({corr_iters} iters) at step {step}")
        
        p_correx = bc_expand_cell(p_corr, BCp)
        grad_p_corr = Grad_cell_to_vels(p_correx)

        u_next = [None, None]
        u_next[0] = u_pred[0] - dt/rho*grad_p_corr[0]
        u_next[1] = u_pred[1] - dt/rho*grad_p_corr[1]
        u_next = bc_apply_vels(u_next, BCu, copy=False)

        if   proj_variant == "non-increment":    p_next = p_corr
        elif proj_variant == "increment":        p_next = pn + p_corr
        elif proj_variant == "increment-rot":    p_next = pn + p_corr - nu*div_u_pred

        unk = u_next
        pnk = p_next

    assert un[0].shape == unk[0].shape
    assert un[1].shape == unk[1].shape
    assert pn.shape == pnk.shape
    return {"u": unk[0], "v": unk[1], "p":pnk, "u_pred": u_pred, "p_corr":p_corr}



cutoff_u = -np.inf
cutoff_p = -np.inf
cutoff_u_post = -np.inf
cutoff_p_post = -np.inf
cutoff_u_corr = -np.inf

import time
def step_phase(
    fields:dict, low_bounds:dict,
    advect_norm:AdvectParams, 
    advect_tang:AdvectParams, 
    step=0,
    use_unmodified_corrector=False,
    proj_variant="increment",
    proj_nonlinear=False,
    proj_iters=1) -> dict:

    # Exact:      
    #   φdu/dt = -φ(u*grad)u + nu*div(φgrad(u)) - φgrad(p)/rho + φS + BCu
    # 
    # Where: 
    #   BCu = -beta/eps^2(1 - φ)(u - g)  //no slip, g is velocity of boundary, so g = 0
    #   div(φgrad(u)) = φlap(u) + grad(u)*grad(φ)
    #   φgrad(p) = grad(φp) - pgrad(φ)  //section 3.1 of [2]
    # 
    # So:
    #   φdu/dt = -φu*div(u) 
    #            + nu*(φlap(u) + grad(u)*grad(φ)) 
    #            - (grad(φp) - pgrad(φ))/rho + φS 
    #            - beta/eps^2(1 - φ)(u - g) 
    #  
    #   du/dt  = -(u*grad)u 
    #            + nu*lap(u) + nu*grad(u)*grad(φ)/φ 
    #            - (grad(φp) - pgrad(φ))/φrho + S 
    #            - beta/eps^2(1 - φ)(u - g)/φ
    # 
    # source: 
    #  [1] https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2009/solving_pdes_in_complex_geometries.pdf?lang=en
    #  [2] https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2010/two_phase_flow.pdf?lang=en
    # 
    # Projection method: 
    #   u -  u      current time step veloctiy
    #   p -  p      current time step pressure
    #   un - u-next next time step velocity, 
    #   pn - p-next next time step pressure, 
    #   us - u-star momentum predictor
    #   q -  p-incr pressure increment (qn = p + q)
    #   NP(v) - non-pressure part of the NS eq. evaluated 
    #           using velocity v (i ommit v below)
    #   
    #   Assume from Helmholtz decomposition that: 
    #       us = un + grad(psi)
    #   since div(un) = 0 (incompressibility) and us is fully general field.
    #   Thus grad(psi) is the error between us and un.
    # 
    #   This gives
    #       (un - u)/dt = NP - grad(pn) 
    #       un = u + dt*NP - dt*grad(p) - dt*grad(q) 
    #       
    #   But we calculate new velocity using opperator splitting using the 
    #   old pressure p, so we get:
    #       (us - u)/dt = NP - grad(p)
    #       us = u + dt*NP - dt*grad(p)
    #   thus and along with the Helmholtz decompositon
    #       us = un + dt*grad(q)
    #       us = un + grad(psi)
    #   we see that psi = dt*q = dt*(pn - p). Thus the increments are
    #       pn = p + q
    #       un = us - dt*grad(q)
    #  
    #   Aapplying div on the Helmholtz decompositon we get 
    #       div(us) = div(un) + div(grad(psi)) 
    #       div(us) = lap(psi) = dt*lap(q)
    #   which yields the poisson eq.
    #       lap(q) = div(us)/dt
    # 
    #   Remarks:
    #    - technically this only works when NP terms are equal.
    #      This happens for explicit schemes where NP in both eqs is NP(u). 
    #      In implicit schemes its NP(us) vs NP(un). We choose to ignore this.
    #    - we get non-incremental by simply assuming p=0 thus pn=q
    #    - to add rho into the procedure replace p with p/rho and q with q/rho
    #   
    #   φdu/dt = duφ/dt - u*dφ/dt
    # 
    #   Now we redo the procedure on our diffuse domain formulation:
    #      φdu/dt = φNP(u) + φgrad(p) + φS
    #      div(φu) = g*grad(φ)
    #   with
    #      NP(v) = -u*div(v) + nu*lap(v) + nu*grad(v)*grad(φ)/φ + BC(v)/φ
    # 
    #   applying the same procdeure as above we get the same result 
    #   (obviously, since we could remove φ from both sides in which case 
    #    the only term that changed is NP which is not present)
    #       φus = φun + dt*φgrad(q)
    #   Now as long as φ != 0 and grad(φ) = 0 we can divide both sides by it and 
    #   retrieve the same possion EQ as above in the classic formulation, 
    #   verifiing it matches Helmholtz decomposition
    #       us = un + dt*grad(q)
    #       us = un + grad(psi).
    # 
    #   This proves that in the liquid domain nothing changed and classic continuity holds. 
    #   In the solid phase or on the boundary we cannot do this so we work with the modified
    #   equation and after applying div() get  
    #       dt*(div(φgrad(q))) = div(φus) - div(φun) = div(φus) - g*grad(φ)
    #   where we used the modified continiuty condition. 
    #
    
    time_start = time.time_ns()
    time_pred = 0
    time_corr = 0
    pred_iters = 0
    corr_iters = 0 

    BCu = [low_bounds['u'], low_bounds['v']]
    BCp = low_bounds['p']
    BCphi = low_bounds['f']

    uxn:Velx = fields['u'] 
    uyn:Vely = fields['v'] 
    pn:Cell  = fields['p'] 
    phi:Cell = fields['f']

    un = bc_apply_vels([uxn, uyn], BCu)
    unk = un
    pnk = pn #todo guess

    # Prepare phase vals ======================
    phim = phi + phi_delta
    ex_phi = bc_expand_cell(phim, BCphi)

    grad_phi_cells = Grad(ex_phi)
    grad_phi = Grad_cell_to_vels(ex_phi)

    vel_phi = Interp_cell_to_vels(ex_phi)
    vel_phi_corner = 1/4*(
        ex_phi[1:,1:] + ex_phi[:-1,1:] +
        ex_phi[1:,:-1] + ex_phi[:-1,:-1]
    )
    # phi on face of particular veclocity cell.
    # So for example phi_vel_face[0] is the phase on the x-normal velocity cell 
    # - phi_vel_face[0][0] is on x-normal face of the x-normal velocity cell 
    # - phi_vel_face[0][1] is on y-normal face of the x-normal velocity cell (ie take faces below each other and iterpolate)
    phi_vel_face = [
        [ex_phi[:,1:-1], vel_phi_corner],
        [vel_phi_corner, ex_phi[1:-1,:]]
    ]

    # Prepare wall velocity and source terms ======================
    wallu = [0, 0]
    vel_wallu = [[0, 0], [0, 0]]
    if wallu_ := fields.get('wallu'):
        wallu = wallu_
        vel_wallu[0] = Interp_cell_to_vels(bc_expand_cell(wallu[0], BCphi))
        vel_wallu[1] = Interp_cell_to_vels(bc_expand_cell(wallu[1], BCphi))

    source = [0, 0]
    vel_source = [[0, 0], [0, 0]]
    if source_ := fields.get('source'):
        source = source_
        vel_source[0] = Interp_cell_to_vels(bc_expand_cell(source[0], BCphi))
        vel_source[1] = Interp_cell_to_vels(bc_expand_cell(source[1], BCphi))

    # It doesnt make sense to do iterations on non-incremental scheme
    if proj_variant == "non-increment":
        proj_iters = 1

    def dot(a, b):
        out = 0
        for (ae, be) in zip(a, b):
            out += ae*be
        return out

    def phase_project(x:np.ndarray, phase:np.ndarray, cutoff, fill = 0, copy=True):
        if copy: x = x.copy()
        if cutoff > -np.inf:
            x = np.where(phase > cutoff, x, fill)
        return x

    p_corr_last = None
    for k in range(proj_iters):
        
        #PREDICTOR ============================
        ex_unknorm = bc_expand_vels(unk if proj_nonlinear else un, BCu)
        #interpolated one field onto the other and expanded
        # according to the others boundary conditons.
        ex_unktang = [ 
            bc_expand_vely(Interp_velx_to_vely(ex_unknorm[0]), BCu[1]),
            bc_expand_velx(Interp_vely_to_velx(ex_unknorm[1]), BCu[0]),
        ]
        
        ex_pnk = bc_expand_cell(pnk, BCp)
        grad_pnk = Grad_cell_to_vels(ex_pnk)
  
        u_pred = [un[0], un[1]]
        for d in range(2):
            t = 1-d
            def pred_lhs(u:np.ndarray) -> np.ndarray:
                nonlocal pred_iters 
                pred_iters += 1

                uex = bc_expand_vel[d](u, BCu[d])

                advnorm = ex_unknorm[d][1:-1, 1:-1]*Advect[d](uex, ex_unknorm[d], advect_norm)
                advtang = ex_unktang[t][1:-1, 1:-1]*Advect[t](uex, ex_unktang[t], advect_tang)
                adv = advnorm + advtang

                dif = nu*DivGrad(phi_vel_face[d], uex)/vel_phi[d]
                BC = -beta/(phi_eps**2) * (1 - vel_phi[d])*(u - vel_wallu[d][d])/vel_phi[d]
                
                lhs = u + dt*(adv - dif - BC)
                lhs = bc_apply_vel[d](lhs, BCu[d], copy=False)
                lhs = phase_project(lhs, vel_phi[d], cutoff_u, copy=False)
                return lhs

            pred_rhs = un[d] + dt*vel_source[d][d]
            if proj_variant != "non-increment": 
                pred_rhs -= dt/rho*grad_pnk[d]

            pred_rhs = phase_project(pred_rhs, vel_phi[d], cutoff_u, copy=False)
            pred_maxiter = max(300, 3 * pred_rhs.size)

            time_pred_start = time.time_ns()
            u_pred[d], pred_iters_ret = matrix_free_solve(pred_lhs, pred_rhs, x0=unk[d], maxiter=pred_maxiter)
            time_pred += time.time_ns() - time_pred_start

            if pred_iters_ret < 0:
                print(f"Predictor ({d=}) breakdown at step {step}")
                return {"u": unk[0], "v": unk[1], "p":pnk, "u_pred": u_pred}
            if pred_iters_ret > 0:
                print(f"Predictor ({d=}) slow convergence ({pred_iters} iters) at step {step}")

        u_pred[0] = phase_project(u_pred[0], vel_phi[0], cutoff_u, fill=vel_wallu[0][0], copy=False)    
        u_pred[1] = phase_project(u_pred[1], vel_phi[1], cutoff_u, fill=vel_wallu[0][1], copy=False)    
        u_pred = bc_apply_vels(u_pred, BCu, copy=False) 

        div_u_pred = Div_vels_to_cell(u_pred)

        #CORRECTOR ============================
        if p_corr_last is None:
            if   proj_variant == "non-increment":    p_corr_last = pnk
            elif proj_variant == "increment":        p_corr_last = None
            elif proj_variant == "increment-rot":    p_corr_last = nu*div_u_pred
 
        corr_maxiter = max(300, 3 * div_u_pred.size)
        time_corr_start = time.time_ns()

        #  dt*(div(grad(q))) = div(φus)
        if use_unmodified_corrector:
            def corr_lhs(q):
                nonlocal corr_iters
                corr_iters += 1
                return Lap(bc_expand_cell(q, BCp))
                
            corr_rhs = rho/dt*div_u_pred
            p_corr, corr_iters_ret = matrix_free_solve(corr_lhs, corr_rhs, x0=p_corr_last, maxiter=corr_maxiter)
        #  dt*(div(φgrad(q))) = div(φus) - g*grad(φ)
        else:
            #The corrector step is where we spend about 90% of runtime therefore its important 
            # to optimize it (within margins). We provide allocation free numpy path
            ex_tmp = bc_expand_cell(pnk, BCp) 
            tmpx = np.empty_like(vel_phi[0])
            tmpy = np.empty_like(vel_phi[1])

            def corr_lhs(q):
                nonlocal corr_iters
                corr_iters += 1

                ex_q = bc_expand_cell(q, BCp, out=ex_tmp)
                lhs = DivGrad(vel_phi, ex_q, scratch_x=tmpx, scratch_y=tmpy)
                return phase_project(lhs, phi, cutoff_p, copy=False)

            div_phi_us = Div_vels_to_cell([vel_phi[0]*u_pred[0], vel_phi[1]*u_pred[1]])
            corr_rhs = rho/dt*(div_phi_us - dot(wallu, grad_phi_cells))
            corr_rhs = phase_project(corr_rhs, phi, cutoff_p, copy=False)

            p_corr, corr_iters_ret = matrix_free_solve(corr_lhs, corr_rhs, x0=p_corr_last, maxiter=corr_maxiter)
            p_corr = phase_project(p_corr, phi, cutoff_p, copy=False)
        
        p_corr_last = p_corr

        if corr_iters_ret < 0:
            print(f"Corrector breakdown at step {step}")
            return {"u": unk[0], "v": unk[1], "p":pnk, "u_pred": u_pred}
        if corr_iters_ret > 0:
            print(f"Corrector slow convergence ({corr_iters} iters) at step {step}")

        time_corr += time.time_ns() - time_pred_start
        
        #UPDATES ============================
        grad_p_corr = Grad_cell_to_vels(bc_expand_cell(p_corr, BCp)) 

        u_next = [None, None]
        u_next[0] = u_pred[0] - phase_project(dt/rho*grad_p_corr[0], vel_phi[0], cutoff_u_corr, copy=False)
        u_next[1] = u_pred[1] - phase_project(dt/rho*grad_p_corr[1], vel_phi[1], cutoff_u_corr, copy=False)

        u_next[0] = phase_project(u_next[0], vel_phi[0], cutoff_u_post, fill=vel_wallu[0][0], copy=False)    
        u_next[1] = phase_project(u_next[1], vel_phi[1], cutoff_u_post, fill=vel_wallu[0][1], copy=False)   
        u_next = bc_apply_vels(u_next, BCu, copy=False)

        if   proj_variant == "non-increment":    p_next = p_corr
        elif proj_variant == "increment":        p_next = pn + p_corr
        elif proj_variant == "increment-rot":    p_next = pn + p_corr - nu*div_u_pred
        p_next = phase_project(p_next, phi, cutoff_p_post, copy=False)

        unk = u_next
        pnk = p_next
    
    time_whole = time.time_ns() - time_start
    print(f"time {time_whole//1e6}ms pred {pred_iters}:{int(time_pred/time_whole*100)}% corr {corr_iters}:{int(time_corr/time_whole*100)}%")

    assert un[0].shape == unk[0].shape
    assert un[1].shape == unk[1].shape
    assert pn.shape == pnk.shape
    return {"u": unk[0], "v": unk[1], "p":pnk, "u_pred": u_pred, "p_corr":p_corr, "f":phi}

def main():
    # PARAMS ======================
    global nx, ny, nu, Lx, Ly, dx, dy, dt
    nx = 160 #num cells
    ny = 60 
    nu = 1.3059e-5 #viscosity
    Ly = 1 #size of domain in meters
    Lx = Ly*nx/ny 
    dx = Lx/nx
    dy = Ly/ny
    dt = 4e-3
    t0 = 0 #begin time
    t1 = 100 #end time
    display_pause = 0 #pause in seconds after each iteration for debugging
    display_every = 10 #update display every X iters. (matplotlib is slow)

    proj_iters = 1 #iterations each time step to minimize splitting error caused by projection method
    proj_variant = "non-increment"
    # proj_variant = "increment"
    # proj_variant = "increment-rot"
    proj_nonlinear = False # Whether to use prev iter or best guess to next iter as the other velocity in advection

    example_fields = False
    # example_fields = True
    phase_field = True
    # phase_field = False
    phase_filed_domain = True
    # phase_filed_domain = False
    
    global phi_width, phi_eps, phi_delta, phi_cutoff 
    phi_delta = 1e-6 #value we add to phi during calculations to regularize the equation in regions where phi=0 
    phi_cutoff = 1e-3 #values under/above 1 minus this are considered pure wall/pure liquid
    phi_width = 6 #width of the phase interface in cells
    phi_eps = calc_phi_eps(phi_width) #width of the phase interface (between phi_cutoff) in real units 
    # phi_width = calc_phi_w(phi_eps)

    # These control when / if the given variable will be projected 
    # onto the phase perscribed value. If not set is not applied.
    # Ie if cutoff_u = 0.2 then in all places where phi < 0.2, u will 
    # be considered a wall thus will be set to the wall velocity (0).
    global cutoff_u, cutoff_p, cutoff_u_post, cutoff_p_post, cutoff_u_corr
    cutoff_u = 0.3 #applied during iterative solve of velocity
    # cutoff_p = 0.2 #applied during iterative solve of pressure
    # cutoff_u_post = 0.2 #applied at the end of the timestep
    # cutoff_p_post = 0.5 #applied at the end of the timestep
    # cutoff_u_corr = 0.5 #applied during corrector update of velocity

    use_unmodified_corrector = False
    # use_unmodified_corrector = True

    parabolic_profile = True

    # domain = {'type':"channel"}
    domain = {'type':"channel", 'circle':True, 'dot_size':0.1*Ly, 'dot_offset':2, 'dot_posx':0.2*Lx, 'dot_posy':0.5*Ly}
    # domain = {'type':"cavity"}
    # domain = {'type':"channel_cavity", "gap":0.2}
    # domain = {'type':"real_cavity"}

    advect_norm = AdvectParams() 
    # advect_norm.variant = "flux"
    advect_norm.variant = "blend"
    advect_norm.factor = 0.95
    advect_norm.dynamic = 0.0
    advect_norm.minblend = 0.0 
    advect_norm.maxblend = 1.0
    
    advect_tang = AdvectParams() 
    # advect_tang.variant = "flux"
    advect_tang.variant = "blend"
    advect_tang.factor = 0.95
    advect_tang.dynamic = 0.0
    advect_tang.minblend = 0.0 
    advect_tang.maxblend = 1.0

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
    # display_field = "phase"
    # display_field = "sdf"

    line_color = "white"
    # line_color = "black"
    display_cell_centers = False
    display_grid = False
    display_BCs = True
    display_phase_walls = True
    display_velocity_arrows = False
    display_face_velocity_arrows = False
    display_face_velocity_arrows_offsets = False
    display_streamlines = False 
    display_streamlines_thickness = False 

    # initial conditions
    boundaries, wall_mask = make_domain(domain, 0, parabolic_profile, phase_filed_domain)
    sdf = -sdf_from_mask(wall_mask)
    phi = sdf_to_phase_field(sdf)
    phi_outline = marching_squares(sdf, 0)
    
    fields = {
        "u": np.zeros((nx+1, ny)),
        "v": np.zeros((nx, ny+1)),
        "p": np.zeros((nx, ny)),
        "f": phi,
        "sdf": sdf,
    }

    # Main loop
    plt.ion()
    fig = plt.figure(figsize=(6*Lx/Ly, 6), dpi=100)
    step = -1
    t = t0 - dt

    #we calculate dt each step to fit the update. 
    # This matters most in the last step where dt is smaller
    normal_dt = dt 
    while t != t1:
        step += 1
        t_old = t
        t = min(t0 + step*normal_dt, t1)
        dt = t - t_old
        
        # BOUNDARIES =========================
        u_in = min(1, t)
        boundaries, wall_mask = make_domain(domain, u_in, parabolic_profile, phase_filed_domain)
        low_bounds = Boundary.to_low_bounds(boundaries.values(), dictify=False)

        # SIMULATE ===========================
        if example_fields:
            new_fields = generate_example_fields()
        else:
            if phase_field:
                new_fields = step_phase(
                    fields, low_bounds,
                    step=step,
                    use_unmodified_corrector=use_unmodified_corrector,
                    proj_variant=proj_variant, 
                    proj_iters=proj_iters, 
                    proj_nonlinear=proj_nonlinear,
                    advect_norm=advect_norm, 
                    advect_tang=advect_tang)
            else:
                new_fields = step(
                    fields, low_bounds,
                    step=step,
                    proj_variant=proj_variant, 
                    proj_iters=proj_iters, 
                    proj_nonlinear=proj_nonlinear,
                    advect_norm=advect_norm, 
                    advect_tang=advect_tang)

        fields.update(new_fields)
        #PLOTTING ============================
        if step % display_every == 0:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_title(f"step = {step} t = {float(t):.6}")
            plot(fig, ax, fields, boundaries, low_bounds,
                display_field=display_field,
                phi_outline = phi_outline,
                line_color = line_color,
                display_phase_walls = display_phase_walls,
                display_cell_centers = display_cell_centers,
                display_grid = display_grid,
                display_BCs = display_BCs,
                display_velocity_arrows = display_velocity_arrows,
                display_face_velocity_arrows = display_face_velocity_arrows,
                display_face_velocity_arrows_offsets = display_face_velocity_arrows_offsets,
                display_streamlines = display_streamlines,
                display_streamlines_thickness = display_streamlines_thickness,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            if display_pause > 0:
                time.sleep(display_pause) 

    plt.ioff()
    plt.show()

def generate_example_fields() -> dict: 
    x_centers = (np.arange(nx) + 0.5) * dx
    y_centers = (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')

    xfu = np.arange(nx + 1) * dx
    yfu = (np.arange(ny) + 0.5) * dy
    Xu, Yu = np.meshgrid(xfu, yfu, indexing='ij')

    xfv = (np.arange(nx) + 0.5) * dx
    yfv = np.arange(ny + 1) * dy
    Xv, Yv = np.meshgrid(xfv, yfv, indexing='ij')

    fields = {}
    fields["u"] = 0.6*np.sin(np.pi * Yu/Ly) + 0.2*(0.5 - Xu/Lx)
    fields["v"] = 0.6*np.cos(np.pi * Xv/Lx) + 0.2*(0.5 - Yv/Ly)
    fields["p"] = np.sin(np.pi * Xc/Lx) * np.cos(np.pi * Yc/Ly)
    return fields

def make_domain(domain:str, u_in:float, parabolic:bool, phase_field:bool) -> Tuple[Boundaries, np.ndarray]:
    def inflow_profile(u:float, n:int) -> np.ndarray:
        if parabolic == False:
            return np.full(n, u)
        centers = (np.arange(n) + 0.5)/n
        profile = u*(1 - (2*centers - 1)**2)
        return profile

    wall = np.zeros((nx, ny), dtype=bool)
    domain_variant = domain.get('type')
    if domain_variant == "channel" and phase_field == False:
        inflow = inflow_profile(u_in, ny)
        boundaries = Boundary.to_dict([
            Boundary.inflow("W", 0, np.arange(ny), inflow, 0),
            Boundary.outflow("E", nx-1, np.arange(ny)),
            Boundary.noslip("S", np.arange(nx), 0),
            Boundary.noslip("N", np.arange(nx), ny-1),
        ])
    elif domain_variant == "cavity" and phase_field == False:
        boundaries = Boundary.to_dict([
            Boundary.noslip("W", 0, np.arange(ny)),
            Boundary.noslip("E", nx-1, np.arange(ny)),
            Boundary.noslip("S", np.arange(nx), 0),
            Boundary.inflow("N", np.arange(nx), ny-1, u_in, 0),
        ])
    elif domain_variant == "real_cavity" and phase_field == False:
        gapW = 3
        gapE = 2
        inflow = inflow_profile(u_in, gapW)
        boundaries = Boundary.to_dict([
            Boundary.inflow("W", 0, np.arange(ny-gapW, ny), inflow, 0),
            Boundary.noslip("W", 0, np.arange(0, ny-gapW)),
            
            Boundary.outflow("E", nx-1, np.arange(ny-gapE, ny)),
            Boundary.slip("E", nx-1, np.arange(0, ny-gapE)),

            Boundary.slip("N", np.arange(nx), ny-1),
            Boundary.noslip("S", np.arange(nx), 0),
        ])

    elif domain_variant == "channel_cavity" and phase_field == False:
        gap = max(ny//5, 1)
        inflow = inflow_profile(u_in, gap)
        boundaries = Boundary.to_dict([
            Boundary.inflow("W", 0, np.arange(ny-gap, ny), inflow, 0),
            Boundary.noslip("W", 0, np.arange(0, ny-gap)),
            Boundary.slip("E", nx-1, np.arange(gap, ny)),
            Boundary.outflow("E", nx-1, np.arange(0, gap)),
            Boundary.noslip("S", np.arange(nx), 0),
            Boundary.noslip("N", np.arange(nx), ny-1),
        ])

    elif domain_variant == "channel" and phase_field == True:
        w = phi_width//2 #wall width
        inflow = inflow_profile(u_in, ny-2*w)
        boundaries = Boundary.to_dict([
            Boundary.noslip("W", 0, np.arange(0, w)),
            Boundary.inflow("W", 0, np.arange(w, ny-w), inflow, 0),
            Boundary.noslip("W", 0, np.arange(ny-w, ny)),
            
            Boundary.noslip("E", nx-1, np.arange(0, w)),
            Boundary.outflow("E", nx-1, np.arange(w, ny-w)),
            Boundary.noslip("E", nx-1, np.arange(ny-w, ny)),

            Boundary.noslip("S", np.arange(nx), 0),
            Boundary.noslip("N", np.arange(nx), ny-1),
        ])

        wall[:, :w] = 1
        wall[:, -w:] = 1
        if 'circle' in domain:
            r   = domain['dot_size']
            px  = domain['dot_posx']
            py  = domain['dot_posy']
            off = domain['dot_offset']
            
            x_centers = (np.arange(nx) + 0.5) * dx
            y_centers = (np.arange(ny) + 0.5) * dy
            Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')
            dist2 = (Xc - px)**2 + (Yc - off*dy - py)**2
            wall[dist2 <= r**2] = 1

    if domain_variant == "channel_cavity" and phase_field == True:
        w = phi_width//2 #wall width
        h = max((ny - 2*w)//5, 1)
        inflow = inflow_profile(u_in, h)

        wall[:w, :] = 1
        wall[-w:, :] = 1
        wall[:, :w] = 1
        wall[:, -w:] = 1
        wall[:w, ny-(h+w):ny-w] = 0
        wall[-w:, w:h+w] = 0
        
        boundaries = Boundary.to_dict([
            Boundary.noslip("W", 0, np.arange(0, ny-(h+w))),
            Boundary.inflow("W", 0, np.arange(ny-(h+w), ny-w), inflow, 0),
            Boundary.noslip("W", 0, np.arange(ny-w, ny)),
            
            Boundary.noslip("E",  nx-1, np.arange(0, w)),
            Boundary.outflow("E", nx-1, np.arange(w, w+h)),
            Boundary.noslip("E",  nx-1, np.arange(w+h, ny)),

            Boundary.noslip("S", np.arange(nx), 0),
            Boundary.noslip("N", np.arange(nx), ny-1),
        ])

    return (boundaries, wall)

def mask_to_boundary_list(mask: np.ndarray):
    m = mask.astype(bool)
    out = {}
    out["E"] = np.where(m[:, :-1] & (~m[:, 1:]))
    out["W"] = np.where(m[:, 1:] & (~m[:, :-1]))
    out["N"] = np.where(m[:-1, :] & (~m[1:, :]))
    out["S"] = np.where(m[1:, :] & (~m[:-1, :]))
    return out

from scipy.ndimage import distance_transform_edt

def sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)

    dist_to_wall = distance_transform_edt(~mask, sampling=(dx, dy))
    dist_to_free = distance_transform_edt(mask, sampling=(dx, dy))

    sdf = dist_to_wall - dist_to_free
    return sdf

def sdf_to_phase_field(sdf:np.ndarray | float) -> np.ndarray | float:
    return 0.5*(1 - np.tanh(3*sdf/phi_eps))

def calc_phi_eps(W:np.ndarray | float) -> np.ndarray | float:
    # calculate transition phi eps such that
    # sdf_to_phase_field(W*min(dx, dy)/2) == phi_cutoff 

    dist = W*min(dx, dy)/2
    eps = 3/np.atanh(1 - 2*phi_cutoff)*dist
    if eps > 0.5:
        eps = 1 - eps
    return eps

def calc_phi_w(phi_eps: float) -> int:
    # np.atanh(1 - 2*delta)/3 = dist/phi_eps
    dist = phi_eps/3*np.atanh(1 - 2*phi_cutoff)
    return int(np.ceil(abs(dist/min(dx, dy)*2)))

from collections import defaultdict
def join_path_segments(segments):
    def first_item(collection):
        if len(collection) == 0:
            return None
        return next(iter(collection))

    if len(segments) == 0:
        return []

    conectivity = defaultdict(set)
    for line in segments:
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])

        conectivity[p1].add(p2)
        conectivity[p2].add(p1)

    lines = []
    while len(conectivity) > 0:
        (p1, set1) = first_item(conectivity.items())
        if len(set1) == 0:
            break

        line = [p1]
        while p2 := first_item(set1):
            set2:set = conectivity[p2]
            line.append(p2)
            
            set1.remove(p2)
            set2.remove(p1)

            if len(set1) == 0: conectivity.pop(p1)
            if len(set2) == 0: conectivity.pop(p2)

            set1 = set2
            p1 = p2

        lines += [np.array(line)]
    return lines

def marching_squares(field, treshold):
    (nx, ny) = field.shape
    discretized = field > treshold
    states = (
        discretized[0:nx-1, 0:ny-1]     # bot left
        + 2*discretized[1:nx,   0:ny-1] # bot right
        + 4*discretized[1:nx,   1:ny]   # top right
        + 8*discretized[0:nx-1, 1:ny]   # top left
    )

    def safe_interp(t, a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(b - a) < 1e-12, 0.5, (t - a) / (b - a))

    fxt_arr = safe_interp(treshold, field[:nx-1, 1:ny], field[1:nx,  1:ny])
    fxb_arr = safe_interp(treshold, field[:nx-1, :ny-1], field[1:nx, :ny-1])
    fyr_arr = safe_interp(treshold, field[1:nx,  :ny-1], field[1:nx,  1:ny])
    fyl_arr = safe_interp(treshold, field[:nx-1, :ny-1], field[:nx-1, 1:ny])

    lines = []
    with_values_mask = np.logical_and(states > 0, states < 15)
    nonzeros = np.nonzero(with_values_mask)
    for x,y in zip(nonzeros[0], nonzeros[1]):
        fxt = fxt_arr[x,y]
        fxb = fxb_arr[x,y]
        fyr = fyr_arr[x,y]
        fyl = fyl_arr[x,y]

        state = states[x,y]

        if state == 1 or state == 14:
            lines.append((x, y+fyl, x+fxb, y))
        elif state == 2 or state == 13:
            lines.append((x+1, y+fyr, x+fxb, y))
        elif state == 3 or state == 12:
            lines.append((x, y+fyl, x+1, y+fyr))
        elif state == 4 or state == 11:
            lines.append((x+fxt, y+1, x+1, y+fyr))
        elif state == 5:
            lines.append((x, y+fyl, x+fxt, y+1))
            lines.append((x+1, y+fyr, x+fxb, y))
        elif state == 6 or state == 9:
            lines.append((x+fxb, y, x+fxt, y+1))
        elif state == 7 or state == 8:
            lines.append((x, y+fyl, x+fxt, y+1))
        elif state == 10:
            lines.append((x, y+fyl, x+fxb, y))
            lines.append((x+fxt, y+1, x+1, y+fyr))
        else:
            assert False 

    lines = np.array(lines)
    return lines


from scipy.ndimage import zoom
from matplotlib.collections import LineCollection
def plot(fig, ax, fields:dict, boundaries:dict, low_bounds:dict, 
    display_field = None,
    phi_outline = None,
    line_color = "white",
    display_phase_walls = False,
    display_cell_centers = False,
    display_grid = False,
    display_BCs = True,
    display_velocity_arrows = False,
    display_face_velocity_arrows = True,
    display_face_velocity_arrows_offsets = False,
    display_streamlines = False,
    display_streamlines_thickness = False,
):
    dd = min(dx, dy)

    ex_p = bc_expand_cell(fields["p"], low_bounds['p'])
    uex = bc_expand_velx(fields["u"], low_bounds['u'])
    vex = bc_expand_vely(fields["v"], low_bounds['v'])

    ax.set_xlim(-dx, (nx + 1) * dx)
    ax.set_ylim(-dy, (ny + 1) * dy)
    ax.set_aspect('equal')

    velx = Interp_vels_to_cellx(uex)
    vely = Interp_vels_to_celly(vex)
    velmag = np.hypot(velx, vely)

    # Field drawing
    display_field_tuple = None
    if   display_field == "p":      display_field_tuple = (ex_p, "pressure")
    elif display_field == "u":      display_field_tuple = (velx, "velocity u")
    elif display_field == "v":      display_field_tuple = (vely, "velocity v")
    elif display_field == "velmag": display_field_tuple = (velmag, "velocity magnitude")
    elif display_field in ["predu", "predv", "predmag"] and "pred" in fields:
        epred = bc_expand_vels(fields["pred"], (BCu, BCv))
        predu = Interp_vels_to_cellx(epred[0]), 
        predv = Interp_vels_to_celly(epred[1])
        if display_field == "predu": display_field_tuple = (predu, "predictor v")
        if display_field == "predv": display_field_tuple = (predv, "predictor u")
        if display_field == "predmag": display_field_tuple = (np.hypot(iepred[0], iepred[1]), "predictor magnitude")
    elif display_field in ["psiu", "psiv", "psimag"]:
        ru = (uex[:-2,1:-1] - uex[1:-1,1:-1])/(uex[1:-1,1:-1] - uex[2:,1:-1])
        rv = (vex[1:-1,:-2] - vex[1:-1,1:-1])/(vex[1:-1,1:-1] - vex[1:-1,2:])

        psiu = np.zeros_like(uex)
        psiu[1:-1, 1:-1] = np.minimum((ru + np.abs(ru)) / (1 + np.abs(ru)), 1)
        
        psiv = np.zeros_like(vex)
        psiv[1:-1, 1:-1] = np.minimum((rv + np.abs(rv)) / (1 + np.abs(rv)), 1)
        
        ipsi = [Interp_vels_to_cellx(psiu), Interp_vels_to_celly(psiv)]
        if display_field == "psiu": display_field_tuple = (psiu, "psiu")
        if display_field == "psiv": display_field_tuple = (psiv, "psiv")
        if display_field == "psimag": display_field_tuple = (np.hypot(ipsi[0], ipsi[1]), "psimag")
    elif display_field == "phase" and "f" in fields:
        display_field_tuple = (bc_expand_cell(fields["f"], low_bounds['f']), "phase")
    elif display_field == "sdf" and "sdf" in fields and "f" in fields:
        display_field_tuple = (bc_expand_cell(fields["sdf"], low_bounds['f']), "phase")

    if display_field_tuple is not None:
        im = ax.imshow(display_field_tuple[0].T, origin='lower', extent=[-dx, (nx+1)*dx, -dy, (ny+1)*dy], interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(display_field_tuple[1])

    x_centers = (np.arange(nx) + 0.5) * dx
    y_centers = (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')

    xfuex = (np.arange(nx+3) - 1) * dx
    yfuex = (np.arange(ny+2) - 1 + 0.5) * dy
    Xuex, Yuex = np.meshgrid(xfuex, yfuex, indexing='ij')

    xfvex = (np.arange(nx+2) - 1 + 0.5) * dx
    yfvex = (np.arange(ny+3) - 1) * dy
    Xvex, Yvex = np.meshgrid(xfvex, yfvex, indexing='ij')

    # grid
    if display_cell_centers:
        ax.scatter(Xc.flatten(), Yc.flatten(), marker='o', color=line_color, s=30)
    if display_grid:
        xfu = np.arange(nx + 1) * dx
        yfu = (np.arange(ny) + 0.5) * dy

        xfv = (np.arange(nx) + 0.5) * dx
        yfv = np.arange(ny + 1) * dy
        ax.plot([xfu, xfu], [np.full(nx+1, 0), np.full(nx+1, Ly)], color=line_color, linewidth=0.4)
        ax.plot([np.full(ny+1, Lx), np.full(ny+1, 0)], [yfv, yfv], color=line_color, linewidth=0.4)

    # velocity arrows
    if display_velocity_arrows:
        scale = np.sqrt(30*30 / velx.size)
        stacked = [Xc, Yc, velx[1:-1,1:-1], vely[1:-1,1:-1]]
        zoomed = [zoom(s, zoom=scale, order=1) if scale < 1 else stacked for s in stacked]
        px, py, vx, vy = zoomed[0], zoomed[1], zoomed[2], zoomed[3]
        ax.quiver(px, py, vx, vy, scale=scale/dd, angles='xy', scale_units='xy', units="dots", color=line_color, width=1)

    # velocity grid
    if display_face_velocity_arrows:
        YuexStaggered = Yuex.copy()
        YuexStaggered[1::2,:] -= 0.05*dy if display_face_velocity_arrows_offsets else 0
        YuexStaggered[0::2,:] += 0.05*dy if display_face_velocity_arrows_offsets else 0

        XvexStaggered = Xvex.copy()
        XvexStaggered[:,1::2] -= 0.05*dx if display_face_velocity_arrows_offsets else 0
        XvexStaggered[:,0::2] += 0.05*dx if display_face_velocity_arrows_offsets else 0
        ax.quiver(Xuex, YuexStaggered, uex, np.zeros_like(uex), scale=1/dd, angles='xy', scale_units='xy', units="dots", color=line_color, width=0.7, headwidth=2, headlength=2)
        ax.quiver(XvexStaggered, Yvex, np.zeros_like(vex), vex, scale=1/dd, angles='xy', scale_units='xy', units="dots", color=line_color, width=0.7, headwidth=2, headlength=2)

    # streamlines
    if display_streamlines:
        lw = 0.8
        density = 1
        if display_streamlines_thickness:
            min_w = 0.2
            max_w = 3
            vel = velmag[1:-1,1:-1]
            lw = (max_w - min_w)*(vel / vel.max()) + min_w
            lw = lw.T
            density = 2
        ax.streamplot(Xc[:,0], Yc[0,:], velx[1:-1,1:-1].T, vely[1:-1,1:-1].T, color=line_color, density=density, linewidth=lw, arrowsize=0.7)

    if display_phase_walls and phi_outline is not None:
        dx_dy = np.array([dx, dy])
        e1 = (phi_outline[:, 0:2] + 0.5) * dx_dy
        e2 = (phi_outline[:, 2:4] + 0.5) * dx_dy
        lc = LineCollection(np.stack([e1, e2], axis=1), colors=line_color, linewidths=1, capstyle="butt")
        ax.add_collection(lc)

    # boundaries
    for b in boundaries.values():
        if display_BCs == False:
            continue

        if b.side == "W": n, t, ddn, ddt, xs, ys = [-1, 0], [0, 1], dx, dy, b.xs, b.ys
        if b.side == "E": n, t, ddn, ddt, xs, ys = [1, 0], [0, -1], dx, dy, b.xs+1, b.ys
        if b.side == "S": n, t, ddn, ddt, xs, ys = [0, -1], [1, 0], dy, dx, b.xs, b.ys
        if b.side == "N": n, t, ddn, ddt, xs, ys = [0, 1], [-1, 0], dy, dx, b.xs, b.ys+1

        n, t = np.array(n), np.array(t)
        coords = np.array([xs, ys]).T
        dx_dy = np.array([dx, dy])
        
        # face edges, face center
        e1 = coords * dx_dy
        e2 = (coords + np.abs(t)) * dx_dy
        ec = (e1 + e2) / 2

        if b.type == "noslip":
            lc = LineCollection(np.stack([e1, e2], axis=1), colors=line_color, linewidths=4, capstyle="butt")
            ax.add_collection(lc)

        if b.type == "slip":
            lc = LineCollection(np.stack([e1, e2], axis=1), colors=line_color, linewidths=4, capstyle="butt", linestyles=":")
            ax.add_collection(lc)

        if b.type == "inflow":
            v = np.vstack((b.valu, b.valv)).T 
            if np.all(v*n == 0):
                soffx, soffy = ec[:,0], ec[:,1]
                if b.side in ["S", "N"]:
                    soffy[0::2] += 0.1*ddn
                    soffy[1::2] += 0.05*ddn
                if b.side in ["W", "E"]:
                    soffx[0::2] += 0.1*ddn
                    soffx[1::2] += 0.05*ddn
                ax.quiver(soffx, soffy, b.valu, b.valv, angles='xy', scale_units='xy', units="dots", color=line_color, scale=1/dd, width=2, minlength=0)
            else:
                offcount = 2
                d = 0.5/offcount
                for off in np.linspace(-0.5+d, 0.5-d, offcount)*ddt:
                    o = ec-v*dd + off*t
                    ax.quiver(o[:,0], o[:,1], b.valu, b.valv, angles='xy', scale_units='xy', units="dots", color=line_color, scale=1/dd, width=2, minlength=0)
            
            lc = LineCollection(np.stack([e1, e2], axis=1), colors=line_color, linewidths=1, capstyle="butt", linestyles=(0, (2, 3)))
            ax.add_collection(lc)

        if b.type == "outflow":
            offsets = [0, 0.2]
            styles = ["-", (0, (2, 3))]
            for style, off in zip(styles, offsets):
                o1 = e1 + off*n*ddn
                o2 = e2 + off*n*ddn

                lc = LineCollection(np.stack([o1, o2], axis=1), colors=line_color, linewidths=1, capstyle="butt", linestyles=style)
                ax.add_collection(lc)

    ax.autoscale()
main()