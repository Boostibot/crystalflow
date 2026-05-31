import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, List, Dict, Literal, Callable, Iterable, Union

R_spec = 287
T = 272 + 20
c_sound = 343

# Predeclared globals. The values are set in the main function
nx, ny, nu, rho, Lx, Ly, dx, dy, dt = 0, 0, 0, 0, 0, 0, 0, 0, 0
rho = 1
beta = 0
phi_width = 0
phi_eps = 0
phi_delta = 0
phi_cutoff = 0

Side         = Literal["N", "S", "E", "W"]
LowBoundType = Literal["val", "der"]
BoundaryType = Literal["inflow", "outflow", "noslip", "slip"]

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
        out = dict()
        for b in bounds:
            name = b.type + b.side
            out[name] = Boundary.concat(out.get(name), b)
        return out
    
    @staticmethod
    def to_low_bounds(bounds:Iterable['Boundary'], dictify=True) -> dict:
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


def bc_expand_velx(f:Velx, bcs:LowBounds, out=None) -> VelxEx:
    if out is None:
        out = np.pad(f, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    else:
        out[1:-1, 1:-1] = f

    for type_side, bc in bcs.items():
        x, y, val = bc.xs+1, bc.ys+1, bc.value
        # directly on the boundary. 
        # Set value *inside* the domain and also expand it one
        # cell out
        if type_side == "valW":   out[x, y] = val; out[x-1, y] = val
        elif type_side == "valE": out[x+1, y] = val; out[x+2, y] = val

        # directly on the boundary, so that central der matches
        elif type_side == "derW": out[x-1, y] = out[x+1, y] - 2*val*dx
        elif type_side == "derE": out[x+2, y] = out[x,   y] + 2*val*dx

        # boundary between cells in tangential direction.
        # Set both ends according to the bc
        elif type_side == "valS": 
            out[x+0, y-1] = 2*val - out[x+0, y]
            out[x+1, y-1] = 2*val - out[x+1, y]
        elif type_side == "valN": 
            out[x+0, y+1] = 2*val - out[x+0, y]
            out[x+1, y+1] = 2*val - out[x+1, y]
        elif type_side == "derS": 
            out[x+0, y-1] = out[x+0, y] - val*dy
            out[x+1, y-1] = out[x+1, y] - val*dy
        elif type_side == "derN": 
            out[x+0, y+1] = out[x+0, y] + val*dy
            out[x+1, y+1] = out[x+1, y] + val*dy

    _bc_fill_corners_pipe(out)
    return out

def bc_expand_vely(f:Vely, bcs:LowBounds, out=None) -> VelyEx:
    if out is None:
        out = np.pad(f, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    else:
        out[1:-1, 1:-1] = f

    for type_side, bc in bcs.items():
        x, y, val = bc.xs+1, bc.ys+1, bc.value
        # directly on the boundary
        if type_side == "valS": out[x, y] = val; out[x, y-1] = val
        elif type_side == "valN": out[x, y+1] = val; out[x, y+2] = val
        # directly on the boundary, so that central der matches
        elif type_side == "derS": out[x, y-1] = out[x, y+1] - 2*val*dy
        elif type_side == "derN": out[x, y+2] = out[x, y  ] + 2*val*dy
        
        # boundary between cells in tangential direction
        elif type_side == "valW": 
            out[x-1, y+0] = 2*val - out[x, y+0]
            out[x-1, y+1] = 2*val - out[x, y+1]
        elif type_side == "valE": 
            out[x+1, y+0] = 2*val - out[x, y+0]
            out[x+1, y+1] = 2*val - out[x, y+1]
        elif type_side == "derW": 
            out[x-1, y+0] = out[x, y+0] - val*dx
            out[x-1, y+1] = out[x, y+1] - val*dx
        elif type_side == "derE": 
            out[x+1, y+0] = out[x, y+0] + val*dx
            out[x+1, y+1] = out[x, y+1] + val*dx

    _bc_fill_corners_pipe(out)
    return out

def bc_expand_cell(f:Cell, bcs:LowBounds, out=None) -> CellEx:
    if out is None:
        out = np.pad(f, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    else:
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
    if bc := bcs.get("valW"): f[bc.xs, bc.ys] = bc.value
    if bc := bcs.get("valE"): f[bc.xs+1, bc.ys] = bc.value
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
    xflat = x0.copy().flat if x0 is not None else None
    xN, iters = sp.sparse.linalg.bicgstab(Aop, Boff, x0=xflat, rtol=rtol, atol=atol, maxiter=maxiter)
    # xN, iters = sp.sparse.linalg.cgs(Aop, Boff, x0=xflat, rtol=rtol, atol=atol, maxiter=maxiter)

    # iters<0: breakdown (solution unreliable)
    if iters < 0:
        return np.zeros_like(b), iters
    # iters==0: converged; iters>0: maxiter hit (partial solution still useful);
    return xN.reshape(b.shape), iters

# Difference operators taking expanded field and returning just the field (eliminates ghost cells)

# interpolate (for interp_velx_to_vely: input x field, output interpolated to y field for interp_vely_to_velx in reverse)
def interp_velx_to_vely(f:VelxEx) -> Vely: 
    # base to match shapes: [1:-1, 1:-1] -> [1:-2,1:]
    # interp between [0, -1], [0, 0], [1, -1], [1, 0] thus the below ranges (add to base shape)
    return 1/4*(f[1:-2,:-1] + f[1:-2,1:] + f[2:-1,:-1] + f[2:-1, 1:])

def interp_vely_to_velx(f:VelyEx) -> Velx: 
    # base to match shapes: [1:-1, 1:-1] -> [1:,1:-2]
    # interp between [-1, 0], [0, 0], [-1, 1], [0, 1] thus the below ranges (add to base shape)
    # should be just transposed interp_velx_to_vely shapes!
    return 1/4*(f[:-1,1:-2] + f[1:,1:-2] + f[:-1,2:-1] + f[1:,2:-1])

# Interpolate velocity field to cell 
def interp_vels_to_cellx(f:VelxEx) -> CellEx: return 1/2*(f[1:,:] + f[:-1,:])
def interp_vels_to_celly(f:VelyEx) -> CellEx: return 1/2*(f[:,1:] + f[:,:-1])
def interp_cell_to_velx(c:CellEx) -> Velx: return 0.5*(c[:-1,1:-1] + c[1:,1:-1])
def interp_cell_to_vely(c:CellEx) -> Vely: return 0.5*(c[1:-1,:-1] + c[1:-1,1:])

def interp_cell_to_vels(c:CellEx) -> List[np.ndarray]: 
    return [interp_cell_to_velx(c), interp_cell_to_vely(c)]

# first derivative
def centralx(u:FieldEx, f=None) -> Field: 
    if f is None:    
        return (u[2:, 1:-1] - u[:-2, 1:-1])*(1/(2*dx))
    uf = u[1:,1:-1] + u[:-1,1:-1]
    prod = uf*(1/(2*dx)*f)
    return np.diff(prod, axis=0)

def centraly(u:FieldEx, f=None) -> Field: 
    if f is None:    
        return (u[1:-1, 2:] - u[1:-1, :-2])*(1/(2*dy))
    uf = u[1:-1,1:] + u[1:-1,:-1]
    prod = uf*(1/(2*dy)*f)
    return np.diff(prod, axis=1)

# first derivative upwind
def upwindx(u, dir_f, f=None): 
    if f is None: f = 1
    out = np.where(dir_f >= 0, u[:-1, 1:-1], u[1:, 1:-1])
    np.multiply(out, f / dx, out=out)
    return out[1:, :] - out[:-1, :]
    
def upwindy(u, dir_f, f=None): 
    if f is None: f = 1
    out = np.where(dir_f >= 0, u[1:-1, :-1], u[1:-1, 1:])
    np.multiply(out, f / dy, out=out)
    return out[:, 1:] - out[:, :-1]

# second derivative
def central2x(f:FieldEx) -> Field: return np.diff(f[:,1:-1], n=2, axis=0) * (1/dx**2)
def central2y(f:FieldEx) -> Field: return np.diff(f[1:-1,:], n=2, axis=1) * (1/dy**2)

def div_gradfx(faces: np.ndarray, u: np.ndarray) -> np.ndarray:
    du = np.diff(u[:, 1:-1], axis=0)   # u[1:,1:-1] - u[:-1,1:-1]
    np.multiply(du, faces, out=du)     # du *= faces
    return np.diff(du, axis=0) * (1/dx**2)

def div_gradfy(faces: np.ndarray, u: np.ndarray) -> np.ndarray:
    du = np.diff(u[1:-1, :], axis=1)   # u[1:-1,1:] - u[1:-1,:-1]
    np.multiply(du, faces, out=du)     # du *= faces
    return np.diff(du, axis=1) * (1/dy**2)

def div_grad(faces:List[np.ndarray], cells:FieldEx) -> Field: 
    return div_gradfx(faces[0], cells) + div_gradfy(faces[1], cells) 

def grad(ex:FieldEx)-> Field:       return [centralx(ex),  centraly(ex)]
def div(ex:List[FieldEx]) -> Field: return centralx(ex[0]) + centraly(ex[1])
def lap(ex:FieldEx) -> Field:       return central2x(ex) + central2y(ex)

def div_vels_to_cell(ex:Vels) -> Cell:
    div = 0 
    div += np.diff(ex[0], n=1, axis=0)/dx
    div += np.diff(ex[1], n=1, axis=1)/dy 
    return div
    
def grad_cell_to_vels(ex:CellEx) -> Vels:
    grad = [None, None]
    grad[0] = np.diff(ex[:,1:-1], n=1, axis=0)/dx
    grad[1] = np.diff(ex[1:-1,:], n=1, axis=1)/dy 
    return grad
    
def _flux_limitx(dd, q, u, f=None, u_face=None, limiter="vanalbada", sweby_beta=1.3):
    qc = q[:, 1:-1]
    uc = u[:, 1:-1]

    if u_face is None: u_face = 0.5 * (uc[:-1, :] + uc[1:, :])
    if f is None: f = 1.0

    assert q.shape == u.shape
    assert u_face.shape[0] == u.shape[0] - 1
    assert u_face.shape[1] == u.shape[1] - 2

    # Van Leer limiter on interior cells only
    du_b = uc[1:-1, :] - uc[:-2, :]
    du_f = uc[2:, :] - uc[1:-1, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.nan_to_num(du_b/du_f, posinf=1e15, neginf=-1e15, nan=0.0, copy=False)

    if limiter == "upwind":
        phi = np.zeros_like(r)
    elif limiter == "vanleer":
        ar = np.abs(r)
        phi = (r + ar) / (1.0 + ar)
    elif limiter == "vanalbada":
        phi = (r*r + r) / (r*r + 1.0)
    elif limiter == "minmod":
        phi = np.maximum(0.0, np.minimum(1.0, r))
    elif limiter == "superbee":
        phi = np.maximum(
            0.0,
            np.maximum(
                np.minimum(2.0 * r, 1.0),
                np.minimum(r, 2.0),
            ),
        )
    elif limiter == "sweby":
        b = sweby_beta
        phi = np.maximum(
            0.0,
            np.maximum(
                np.minimum(b * r, 1.0),
                np.minimum(r, b),
            ),
        )
    elif limiter == "mc":
        phi = np.maximum(
            0, np.minimum.reduce([0.5*(1+r), 2*np.ones_like(r),2*r])
        )
        
    else:
        raise ValueError(f"Unknown limiter '{limiter}'")


    # Pad the slope with zeros. This has the effect of
    # falling back onto first-order upwind in the missing cells! 
    slope = np.zeros_like(qc)
    slope[1:-1, :] = phi*(qc[1:-1, :] - qc[:-2, :])

    qL = qc[:-1, :] + 0.5 * slope[:-1, :]
    qR = qc[1:,  :] - 0.5 * slope[1:,  :]

    q_up = np.where(u_face >= 0.0, qL, qR)
    F = q_up * (f / dd)
    return F[1:, :] - F[:-1, :]

def flux_limitx(q, u, f=None, u_face=None, limiter="vanalbada", sweby_beta=1.3) -> Velx:
    return _flux_limitx(dx, q, u, u_face=u_face, f=f, limiter=limiter, sweby_beta=sweby_beta)
    
def flux_limity(q, u, f=None, u_face=None, limiter="vanalbada", sweby_beta=1.3) -> Vely:
    fT = f.T if f is not None else None
    u_faceT = u_face.T if u_face is not None else None
    return _flux_limitx(dy, q.T, u.T, u_face=u_faceT, f=fT, limiter=limiter, sweby_beta=sweby_beta).T

interp_vels_swap  = (interp_velx_to_vely,  interp_vely_to_velx)
interp_vels_to_cell = (interp_vels_to_cellx, interp_vels_to_celly)
central = (centralx,  centraly)
central2 = (central2x, central2y)
upwind = (upwindx, upwindy)
flux_limit = (flux_limitx, flux_limity)

LimiterType = Literal[
    "central",
    "upwind",
    "blend",
    "vanleer",
    "vanalbada",
    "minmod",
    "superbee",
    "sweby",
    "mc",
]
AdvectionVariant = Literal["advect", "div", "skew", "non-conservative"]

@dataclass
class AdvectParams:
    limiter : LimiterType  = "vanalbada"
    # see "Fully Conservative Higher Order Finite Difference Schemes for Incompressible Flow" by
    # "Y. Morinishi,1 T. S. Lund, O. V. Vasilyev, and P. Moin" 1998
    variant : AdvectionVariant = "advect" 
    blend_factor : float = 0.0 #Applies only when limiter is blend. 0 only upwind to 1 only central difference
    sweby_beta : float = 1.5 #Applies only when limiter is sweby

# Calculates axis component of the mathematical expression "u*grad(q)" 
# in few different ways depending on params. Is intended to be used as catch-all advection solution 
# - q is the advected field (arbitrary)
# - u is the advecting field (velocity)
# - u_face are the faces of u. If not provided is calculated from u which might however be worse approximation.
# this function is linear in q. All nonlinear behaviour is based on u / u_face.
def advectxy(axis, q:np.ndarray, u:np.ndarray, params:AdvectParams, u_face:np.ndarray=None) -> np.ndarray:
    def calc_face(field):
        if axis == 0: return 0.5*(field[1:,1:-1] + field[:-1,1:-1])
        if axis == 1: return 0.5*(field[1:-1,1:] + field[1:-1,:-1])
    
    if u_face is None: u_face = calc_face(u)
    
    # f here and in all other functions (upwind, central) is either u_face or nothing
    def _advect_limiter(q:np.ndarray, f=None):
        if params.limiter in ["vanleer", "vanalbada", "minmod", "superbee", "sweby", "mc"]:
            return flux_limit[axis](q, u, u_face=u_face, f=f, limiter=params.limiter, sweby_beta=params.sweby_beta)
        elif params.limiter == "central":
            return central[axis](q, f=f)
        elif params.limiter == "upwind":
            return upwind[axis](q, u_face, f=f)
        elif params.limiter == "blend":
            upw = upwind[axis](q, u_face, f=f)
            cen = central[axis](q, f=f)
            return upw + params.blend_factor*(cen - upw)
        else:
            raise ValueError(f"Unknown flux limiter '{params.limiter}'")

    if params.variant == "advect":
        return _advect_limiter(q, f=u_face)
    #TODO: this one only holds with the classic continuity EQ. Make work with the modified one!
    elif params.variant == "div": 
        return _advect_limiter(u*q)
    elif params.variant == "skew":
        div = _advect_limiter(u*q)
        adv = _advect_limiter(q, f=u_face)
        return 0.5*(div + adv)
    elif params.variant == "non-conservative":
        return u[1:-1, 1:-1]*_advect_limiter(q)
    else:
        raise ValueError(f"Unknown advection varaint '{params.variant}'")
    
def advectx(q:np.ndarray, u:np.ndarray, params:AdvectParams, u_face=None) -> np.ndarray: 
    return advectxy(0, q, u, params, u_face=u_face)
def advecty(q:np.ndarray, u:np.ndarray, params:AdvectParams, u_face=None) -> np.ndarray: 
    return advectxy(1, q, u, params, u_face=u_face)

advect = (advectx, advecty)

# DIRECT MATRIX ASSEMBLY 
def mat_rows(shape, ce=0, px=0, mx=0, py=0, my=0, offset=0):
    return mat_from_row(shape, mat_row(ce, px, mx, py, my, offset))

def mat_cols(shape, ce=0, px=0, mx=0, py=0, my=0, offset=0):
    ce = np.broadcast_to(ce, shape)
    px = np.broadcast_to(px, shape)
    mx = np.broadcast_to(mx, shape)
    py = np.broadcast_to(py, shape)
    my = np.broadcast_to(my, shape)
    offset = np.broadcast_to(offset, shape)
    return np.stack([ce, px, mx, py, my, offset], axis=-1,)

def mat_diag(shape):
    return mat_cols(shape, ce=np.broadcast_to(1, shape))

def mat_off(u, shape=None):
    return mat_cols(u.shape if shape is None else shape, offset=u)

def mat_scale(u):
    return np.broadcast_to(u[..., None], (*u.shape, 6))

def mat_row(ce=0, px=0, mx=0, py=0, my=0, offset=0):
    return np.array([ce, px, mx, py, my, offset])
    
def mat_from_row(shape, row):
    return np.broadcast_to(row, (*shape, len(row))) 

def mat_centralx(shape, f=None): 
    if f is None: return mat_from_row(shape, mat_row(px=1, mx=-1)/(2*dx)) 
    return mat_cols(shape, px=f[1:, :], ce=-f[:-1, :] + f[1:, :], mx=-f[:-1, :]) / (2*dx)

def mat_centraly(shape, f=None): 
    if f is None: return mat_from_row(shape, mat_row(py=1, my=-1)/(2*dy)) 
    return mat_cols(shape, py=f[:,1:], ce=-f[:,:-1] + f[:,1:], my=-f[:,:-1]) / (2*dy)

def mat_central2x(shape): return mat_from_row(shape, mat_row(px=1, ce=-2, mx=1)/dx**2) 
def mat_central2y(shape): return mat_from_row(shape, mat_row(py=1, ce=-2, my=1)/dy**2) 

def mat_grad(shape): return [mat_centralx(shape), mat_centraly(shape)]
def mat_lap(shape): return mat_central2x(shape) + mat_central2y(shape)

def mat_div_gradx(shape, f): return mat_cols(shape, mx=f[:-1, :], ce=-(f[:-1, :] + f[1:, :]), px=f[1:, :]) / dx**2
def mat_div_grady(shape, f): return mat_cols(shape, my=f[:, :-1], ce=-(f[:, :-1] + f[:, 1:]), py=f[:, 1:]) / dy**2
def mat_div_grad(shape, fs): return mat_div_gradx(shape, fs[0]) + mat_div_grady(shape, fs[1])

def mat_upwindx(shape, dir, f=None):
    # u > 0: (ui - ui-1) * fi-1/2 / dx (f[:-1])
    # else:  (ui+1 - ui) * fi+1/2 / dx (f[1:])
    if f is None: f = 1
    fdx = f / dx
    if isinstance(f, np.ndarray):
        fp = fdx[1:, :]
        fm = fdx[:-1, :]
    else:
        fp = fdx 
        fm = fdx

    mask = dir >= 0
    mx = np.where(mask, -fm, 0.0)
    ce = np.where(mask, fm, -fp)
    px = np.where(mask, 0.0, fp)
    return mat_cols(shape, ce=ce, px=px, mx=mx)
    
def mat_upwindy(shape, dir, f=None):
    if f is None: f = 1
    fdy = f / dy
    if isinstance(f, np.ndarray):
        fp = fdy[:, 1:]
        fm = fdy[:, :-1]
    else:
        fp = fdy
        fm = fdy

    mask = dir >= 0
    my = np.where(mask, -fm, 0.0)
    ce = np.where(mask, fm, -fp)
    py = np.where(mask, 0.0, fp)
    return mat_cols(shape, ce=ce, py=py, my=my)

# Advection, for now limited to blend... 
def mat_advectx(shape, dir, params:AdvectParams, f:np.ndarray=None) -> np.ndarray:
    upw = mat_upwindx(shape, dir, f)
    cen = mat_centralx(shape, f)
    return params.blend_factor*cen + (1 - params.blend_factor)*upw

def mat_advecty(shape, dir, params:AdvectParams, f:np.ndarray=None) -> np.ndarray:
    upw = mat_upwindy(shape, dir, f)
    cen = mat_centraly(shape, f)
    return params.blend_factor*cen + (1 - params.blend_factor)*upw

mat_central = (mat_centralx, mat_centraly)
mat_upwind = (mat_upwindx, mat_upwindy)
mat_advect = (mat_advectx, mat_advecty)

# "Pins" the stencil at particular xs, ys to the perscribed value.
# This has two modes: 
# direct = True: which causes mat_stencil_apply(stencils, .)[xs, ys] == value
# direct = False: which causes the solution of this stencil to have the value:
#    A,b = mat_stencil_to_dia(stencils)
#    Ax = -b #solve for x
#    x[xs, ys] = value
def mat_pin(stencils: np.ndarray, xs, ys, value, direct=False):
    stencils[xs, ys, :] = 0
    if direct:
        stencils[xs, ys, 0] = 0
        stencils[xs, ys, 5] = value
    else:
        stencils[xs, ys, 0] = 1
        stencils[xs, ys, 5] = -value

def mat_apply_cell_bcs(stencils: np.ndarray, bcs:dict, copy=True) -> np.ndarray:
    CE, PX, MX, PY, MY, OFF = range(6)
    out = np.array(stencils, copy=copy)

    def _apply(xs, ys, coeff_idx, sign_center, value, scale):
        coeff = out[xs, ys, coeff_idx]
        out[xs, ys, CE] += sign_center * coeff
        out[xs, ys, OFF] += scale * value * coeff
        out[xs, ys, coeff_idx] = 0

    for side, bc in bcs.items():
        xs, ys, val = bc.xs, bc.ys, bc.value

        if side == "valW":   _apply(xs, ys, MX, -1.0, val, 2.0) # u[-1,j] = 2g - u[0,j]
        elif side == "valE": _apply(xs, ys, PX, -1.0, val, 2.0) # u[nx,j] = 2g - u[nx-1,j]
        elif side == "valS": _apply(xs, ys, MY, -1.0, val, 2.0) # u[i,-1] = 2g - u[i,0]
        elif side == "valN": _apply(xs, ys, PY, -1.0, val, 2.0) # u[i,ny] = 2g - u[i,ny-1]
        elif side == "derW": _apply(xs, ys, MX, +1.0, val, -dx) # u[-1,j] = u[0,j] - g*dx
        elif side == "derE": _apply(xs, ys, PX, +1.0, val, +dx) # u[nx,j] = u[nx-1,j] + g*dx
        elif side == "derS": _apply(xs, ys, MY, +1.0, val, -dy) # u[i,-1] = u[i,0] - g*dy
        elif side == "derN": _apply(xs, ys, PY, +1.0, val, +dy) # u[i,ny] = u[i,ny-1] + g*dy
        else: raise ValueError(f"Unknown boundary type: {side}")

    return out

def mat_apply_velxy_bcs(stencils: np.ndarray, bcs: dict, is_y: bool, copy=True, direct=False) -> np.ndarray:
    CE, PX, MX, PY, MY, OFF = range(6)
    out = np.array(stencils, copy=copy)

    def _apply(xs, ys, src_idx, dst_scale, dst_idx, value):
        coeff = out[xs, ys, src_idx]
        out[xs, ys, dst_idx] += dst_scale*coeff
        out[xs, ys, OFF] += value * coeff
        out[xs, ys, src_idx] = 0

    for side, bc in bcs.items():
        xs, ys, val = bc.xs, bc.ys, bc.value

        if not is_y:
            if side == "valW":   mat_pin(out, xs+0, ys, val, direct=direct)
            elif side == "valE": mat_pin(out, xs+1, ys, val, direct=direct)
            elif side == "valS": _apply(xs, ys, MY, -1.0, CE, val * 2.0)        
            elif side == "valN": _apply(xs, ys, PY, -1.0, CE, val * 2.0)        
            elif side == "derW": _apply(xs+0, ys, MX, +1.0, PX, val * -2.0 * dx) 
            elif side == "derE": _apply(xs+1, ys, PX, +1.0, MX, val * +2.0 * dx)
            elif side == "derS": _apply(xs, ys, MY, +1.0, CE, val * -dy)
            elif side == "derN": _apply(xs, ys, PY, +1.0, CE, val * +dy)
            else: raise ValueError(f"Unknown boundary type: {side}")
        else:
            if side == "valW":   _apply(xs, ys, MX, -1.0, CE, val * 2.0) 
            elif side == "valE": _apply(xs, ys, PX, -1.0, CE, val * 2.0)
            elif side == "valS": mat_pin(out, xs, ys+0, val, direct=direct)
            elif side == "valN": mat_pin(out, xs, ys+1, val, direct=direct)
            elif side == "derW": _apply(xs, ys, MX, +1.0, CE, val * -dx)
            elif side == "derE": _apply(xs, ys, PX, +1.0, CE, val * +dx)
            elif side == "derS": _apply(xs, ys+0, MY, +1.0, PY, val * -2.0 * dy)  
            elif side == "derN": _apply(xs, ys+1, PY, +1.0, MY, val * +2.0 * dy)
            else: raise ValueError(f"Unknown boundary type: {side}")

    return out

def mat_apply_velx_bcs(stencils: np.ndarray, bcs: dict, copy=True, direct=False) -> np.ndarray:
    return mat_apply_velxy_bcs(stencils, bcs, is_y=False, copy=copy, direct=direct)
    
def mat_apply_vely_bcs(stencils: np.ndarray, bcs: dict, copy=True, direct=False) -> np.ndarray:
    return mat_apply_velxy_bcs(stencils, bcs, is_y=True, copy=copy, direct=direct)

mat_apply_vel_bcs = (mat_apply_velx_bcs, mat_apply_vely_bcs)

def mat_stencil_apply(st: np.ndarray, u: np.ndarray) -> np.ndarray:
    assert u.ndim == 2 and st.ndim == 3 and st.shape[-1] == 6
    assert u.shape == st.shape[:2]

    up = np.pad(u, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    CE, PX, MX, PY, MY, OFF = range(6)
    out = (
        st[..., OFF]
        + st[..., CE] * up[1:-1, 1:-1]
        + st[..., PX] * up[2:, 1:-1]
        + st[..., MX] * up[:-2, 1:-1]
        + st[..., PY] * up[1:-1, 2:]
        + st[..., MY] * up[1:-1, :-2]
    )
    return out

def mat_stencil_to_dia(stencils: np.ndarray):
    assert stencils.ndim == 3 and stencils.shape[-1] == 6
    nx, ny, _ = stencils.shape
    n = nx * ny

    CE, PX, MX, PY, MY, OFF = range(6)
    ce = stencils[..., CE].reshape(-1)
    px = stencils[..., PX].reshape(-1)
    mx = stencils[..., MX].reshape(-1)
    py = stencils[..., PY].reshape(-1)
    my = stencils[..., MY].reshape(-1)
    b = stencils[..., OFF].reshape(-1)

    d_py = py[:-1].copy()   # offset +1 has length n-1
    d_my = my[1:].copy()    # offset -1 has length n-1

    bad = np.arange(ny - 1, n - 1, ny)  # row-end positions in the flattened grid
    d_py[bad] = 0.0
    d_my[bad] = 0.0

    d_px = px[:-ny]
    d_mx = mx[ny:] 

    A = sp.sparse.diags_array(
        [ce, d_py, d_my, d_px, d_mx],
        offsets=[0, 1, -1, ny, -ny],
        shape=(n, n),
        format="dia"
    )
    return A, b

def ilu_preconditioner(A, drop_tol=1e-4, fill_factor=10):
    A = A.tocsc()
    ilu = sp.sparse.linalg.spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
    M = sp.sparse.linalg.LinearOperator(A.shape, matvec=ilu.solve)
    return M

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

def freeze(xs):
    if isinstance(xs, (list, tuple)):
        for x in xs:
            freeze(x)
    elif isinstance(xs, np.ndarray):
        xs.flags.writeable = False
    return xs

#TODO: move more stuff into params, less as globals (just model params. Impl params will be arguments)
step_total_time_sum = 0
step_pred_time_sum = 0
step_corr_time_sum = 0

M = None

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
# source: 
#  [1] https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2009/solving_pdes_in_complex_geometries.pdf?lang=en
#  [2] https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2010/two_phase_flow.pdf?lang=en
# 
# Projection method: 
#   u -  u      current time step veloctiy
#   p -  p      current time step pressure
#   pp - p-pred predicted next time step pressure (may be p)
#   um - u-next next time step velocity, 
#   pm - p-next next time step pressure, 
#   us - u-star momentum predictor
#   dp -  p-incr pressure increment from predicted (pm = pp + dp)
#   NP(v) - non-pressure part of the NS eq. evaluated 
#           using velocity v (i ommit v below)
#   
#   Assume from Helmholtz decomposition that: 
#       us = um + grad(psi)
#   since div(um) = 0 (incompressibility) and us is fully general field.
#   Thus grad(psi) is the error between us and um.
# 
#   Our equation reads:
#       du/dt = -(u*grad)u + nu*div(grad(u)) + S - grad(p)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                   NP (non pressure terms)    
#       du/dt = NP - grad(p)  
# 
#   We discretize using crank-nicolson with cofficient a=0.5 so we get           
#       (um - u)/dt = a[NP - grad(pm)] + (1-a)[NP - grad(p)]
#       um = u + dt*NP - a*dt*grad(pm = pp+dp) - (1-a)*dt*grad(p) 
#       
#   But we can calculate the RHS only using our best guess for pressure pp 
#   thus we drop the dp and get:
#       us = u + dt*NP - a*dt*grad(pp) - (1-a)*dt*grad(p) 
#   and can write
#       us = um + a*dt*grad(dp)
#   and along with the Helmholtz decompositon
#       us = um + grad(psi)
#   we see that psi = a*dt*dp = a*dt*(pm - pp). Thus the increments are
#       pm = pp + dp
#       um = us - a*dt*grad(dp)
#  
#   Aapplying div on the increment we have
#       div(us) = a*dt*lap(dp)
#   which yields the poisson eq.
#       lap(dp) = div(us)/(dt*a)
# 
#   Remarks:
#    - technically this only works when NP terms are equal.
#      This happens for explicit schemes where NP in both eqs is NP(u). 
#      In implicit schemes its NP(us) vs NP(um). We choose to ignore this.
#    - we get non-incremental by simply assuming p=0 thus pm=dp
#    - to add rho into the procedure replace p with p/rho and dp with dp/rho
#
#   Now we redo the procedure on our diffuse domain formulation:
#      φdu/dt = φNP(u) - φgrad(p)
#      div(φu) = g*grad(φ)
#   with
#      NP(v) = -u*div(v) + nu*lap(v) + nu*grad(v)*grad(φ)/φ + BC(v)/φ + S
# 
#   applying the same procdeure as above we get the same result 
#   (obviously, since we could remove φ from both sides in which case 
#    the only term that changed is NP which is not present)
#       φus = φum + a*dt*φgrad(dp)
#   Now as long as φ != 0 and grad(φ) = 0 we can divide both sides by it and 
#   retrieve the same possion EQ as above in the classic formulation, 
#   verifiing it matches Helmholtz decomposition
#       us = um + a*dt*grad(dp)
#       us = um + grad(psi).
# 
#   This shows that in the liquid domain nothing changed and classic continuity holds. 
#   In the solid phase or on the boundary we cannot do this so we work with the modified
#   equation and after applying div() get  
#       a*dt*(div(φgrad(dp))) = div(φus) - div(φum) = div(φus) - g*grad(φ)
#   where we used the modified continiuty condition. 
#       
def step_phase(
    fields:dict, low_bounds:dict,
    advect_params:AdvectParams | None = None, 
    past_fields : List[dict] | None = None,
    
    proj_variant = "increment",
    proj_outer_iters = 1,

    proj_bdf_order = 2,
    proj_extrapolate_u = 2, 
    proj_extrapolate_p = 2, 

    proj_pred_maxiter = 50,
    proj_corr_maxiter = 1000,
    
    proj_phase_cutoff = 0.05,

    proj_preconditioner_every = 30,
    proj_preconditioner_fill_factor = 35,

    proj_nonlinear_predictor = False,
    proj_phase_corrector = True,
    proj_corr_p_bcs = False,
    
    
    step = 0) -> dict:

    if advect_params is None:
        advect_params = AdvectParams()

    # Perf info 
    time_start = time.time_ns()
    time_pred = 0
    time_corr = 0
    time_pred_solve = 0
    time_corr_solve = 0
    pred_iters = 0
    corr_iters = 0 

    BCu = [low_bounds['u'], low_bounds['v']]
    BCp = low_bounds['p']
    BCphi = low_bounds['f']

    uxn:Velx = freeze(fields['u']) 
    uyn:Vely = freeze(fields['v']) 
    pn:Cell  = freeze(fields['p'])
    phi:Cell = freeze(fields['f'])

    un = freeze(bc_apply_vels([uxn, uyn], BCu))

    # Prepare phase ======================
    phim = phi + phi_delta
    ex_phi = bc_expand_cell(phim, BCphi)

    grad_phi_cells = grad(ex_phi)
    grad_phi = grad_cell_to_vels(ex_phi)

    vel_phi = interp_cell_to_vels(ex_phi)
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
    # TODO: phi_vel_face can be tighteneded I think

    # Prepare wall velocity and source terms ======================
    wallu = [0, 0]
    vel_wallu = [[0, 0], [0, 0]]
    if w := fields.get('wallu'):
        wallu = w
        vel_wallu[0] = interp_cell_to_vels(bc_expand_cell(wallu[0], BCphi))
        vel_wallu[1] = interp_cell_to_vels(bc_expand_cell(wallu[1], BCphi))

    source = [0, 0]
    vel_source = [[0, 0], [0, 0]]
    if s := fields.get('source'):
        source = s
        vel_source[0] = interp_cell_to_vels(bc_expand_cell(source[0], BCphi))
        vel_source[1] = interp_cell_to_vels(bc_expand_cell(source[1], BCphi))

    if proj_phase_corrector or proj_corr_p_bcs:
        BC_corr = BCp
    else:
        BC_corr = {
            "derW": LowBound("W", "der", 0, np.arange(ny), 0),
            "derE": LowBound("E", "der", nx-1, np.arange(ny), 0),
            "derS": LowBound("S", "der", np.arange(nx), 0, 0),
            "derN": LowBound("N", "der", np.arange(nx), ny-1, 0),
        }

    # Below is implemented the BDFq time integration method.
    # See: "An overview of projection methods for incompressible flows"
    # https://www.math.purdue.edu/~shen7/pub/remarks_revised.pdf
    # 
    #   1. 1/dt[ us_coeff*us - rest(un) ] = Dif(us) - Adv(us, up) - grad(pp) + BC(us)  + S
    # 
    #   2. us_coeff/dt[ um - us] + grad(dp) = 0
    # 
    #   3. pm = pn + dp - nu*div(us)  
    # 
    # Where: 
    #    us_coeff*us - rest(un) = "extrapolate us from un, un-1, ..." - un 
    # us_coeff is simply the coefficient in front of us in this expression and
    # rest is the rest of the terms
    # 
    # Thus the final solution is:
    #  1. us - dt/us_coeff[ Dif(us) - Adv(us, up) + BC(us) ] = 1/us_coeff*rest(un) + dt/us_coeff[ -grad(pp) + S ]
    #                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                        ^^^^^^^^^^
    #                                LIN                                                                FLAT
    #  2. lap(dp) = us_coeff/dt * div(us)
    # 
    #  3. um = us - dt/us_coeff * grad(dp)
    #     pm = pn + dp - nu*div(us)  
    # 
    # Where LIN, FLAT are just functions for ease of implementation (or rather ease of changing schemes)

    # 1/dt[ us_coeff*us - rest(un) ] = -grad(pp)
    # lap(dp) = us_coeff/dt * div(us)
    proj_bdf_order = min(proj_bdf_order, len(past_fields)) 
    proj_extrapolate_u = min(proj_extrapolate_u, len(past_fields)) 
    proj_extrapolate_p = min(proj_extrapolate_p, len(past_fields)) 

    # returns (us_coeff, rest(field))
    def BDFq(order, field) -> Tuple[float, np.ndarray]:
        if order == 2:
            return (3/2, 2*past_fields[0][field] - 1/2*past_fields[1][field])
        elif order == 1:
            return (1, past_fields[0][field])
        else:
            raise ValueError(f"Invalid BDFq order {order} for field '{field}'")

    us_coeff, rest_unx = BDFq(proj_bdf_order, "u")
    us_coeff, rest_uny = BDFq(proj_bdf_order, "v")
    rest_un = [rest_unx, rest_uny]

    # extrapolate previous solutions. This is used as initial guess for matrix solve
    # and depending on other settings during non-linear advection causing for faster 
    # convergence across outer iterations
    def extrapolate(order, field):
        order = min(order, len(past_fields))
        if order == 3:
            return 3*past_fields[0][field] - 3*past_fields[1][field] + past_fields[2][field]
        elif order == 2:
            return 2*past_fields[0][field] - past_fields[1][field]
        elif order == 1:
            return past_fields[0][field]
        else:
            raise ValueError(f"Invalid extrapolation order {order} for field '{field}'")

    um_extrapolated = [0, 0]
    um_extrapolated[0] = extrapolate(proj_extrapolate_u, "u")
    um_extrapolated[1] = extrapolate(proj_extrapolate_u, "v")
    pm_extrapolated    = extrapolate(proj_extrapolate_p, "p")

    freeze(rest_un)
    freeze(um_extrapolated)
    freeze(pm_extrapolated)

    dp_last = None
    umk = None
    pmk = None
    for k in range(proj_outer_iters):
        # This function prepares fields for LIN and FLAT to remove repeated computation
        def compute_interpolated(u, p) -> tuple:
            ex_unorm = bc_expand_vels(u, BCu)
            ex_utang = [ 
                bc_expand_vely(interp_velx_to_vely(ex_unorm[0]), BCu[1]),
                bc_expand_velx(interp_vely_to_velx(ex_unorm[1]), BCu[0]),
            ]
            cell_u = [
                interp_vels_to_cellx(ex_unorm[0]),
                interp_vels_to_celly(ex_unorm[1]),
            ]
            face_unorm = [
                cell_u[0][:, 1:-1],
                cell_u[1][1:-1, :],
            ]
            face_utang = [
                0.5*(ex_unorm[0][1:-1, 1:] + ex_unorm[0][1:-1, :-1]),
                0.5*(ex_unorm[1][1:, 1:-1] + ex_unorm[1][:-1, 1:-1]),
            ]
            
            ex_p = bc_expand_cell(p, BC_corr)
            grad_p = grad_cell_to_vels(ex_p)

            return freeze((ex_unorm, ex_utang, face_unorm, face_utang, grad_p))
        
        def LIN(uex:np.ndarray, d:int, t:int, interpolated:tuple) -> np.ndarray:
            ex_unorm, ex_utang, face_unorm, face_utang, grad_p = interpolated
            advnorm = advect[d](uex, u=ex_unorm[d], u_face=face_unorm[d], params=advect_params)
            advtang = advect[t](uex, u=ex_utang[t], u_face=face_utang[t], params=advect_params)
            adv = (advnorm + advtang)

            dif = nu*div_grad(phi_vel_face[d], uex)/vel_phi[d]
            BC = -beta/(phi_eps**2) * (1 - vel_phi[d])*(uex[1:-1, 1:-1] - vel_wallu[d][d])/vel_phi[d]

            return -adv + dif + BC

        def FLAT(d:int, t:int, interpolated:tuple) -> np.ndarray:
            ex_unorm, ex_utang, face_unorm, face_utang, grad_p = interpolated
            out = vel_source[d][d]
            if proj_variant != "non-increment": 
                out -= 1/rho*grad_p[d]
            return out

        #PREDICTOR ============================
        time_pred_start = time.time_ns()

        # Predicted variables are used to represent 
        # next time time level in predictor solve
        if k == 0:
            um_predicted = um_extrapolated if proj_nonlinear_predictor else un
            pm_predicted = pm_extrapolated
        else:
            um_predicted = umk
            pm_predicted = pmk
        
        # initial guess for solution. 
        # This can be arbitrary but can speed up iteration
        um_guess = um_extrapolated if k == 0 else umk

        interp_m = compute_interpolated(um_predicted, pm_predicted)

        us = list(um_predicted)
        for d in range(2):
            t = 1-d
            def pred_lhs(u:np.ndarray) -> np.ndarray:
                nonlocal pred_iters 
                pred_iters += 1
                uex = bc_expand_vel[d](u, BCu[d])
                out = u - dt/us_coeff*LIN(uex, d, t, interp_m)
                out = bc_apply_vel[d](out, BCu[d], copy=False)
                out = phase_project(out, vel_phi[d], proj_phase_cutoff, copy=False)
                return out
                
            pred_rhs = 1/us_coeff*rest_un[d] + dt/us_coeff*FLAT(d, t, interp_m) 
            pred_rhs = bc_apply_vel[d](pred_rhs, BCu[d], copy=False)
            pred_rhs = phase_project(pred_rhs, vel_phi[d], proj_phase_cutoff, copy=False)

            time_pred_solve_start = time.time_ns()
            us[d], pred_iters_ret = matrix_free_solve(pred_lhs, pred_rhs, x0=um_guess[d], maxiter=proj_pred_maxiter)
            time_pred_solve += time.time_ns() - time_pred_solve_start

            if pred_iters_ret < 0: print(f"Predictor breakdown at step {step}"); return {}
            if pred_iters_ret > 0: print(f"Predictor slow convergence ({pred_iters} iters) at step {step}")

        us[0] = phase_project(us[0], vel_phi[0], proj_phase_cutoff, copy=False)    
        us[1] = phase_project(us[1], vel_phi[1], proj_phase_cutoff, copy=False)    
        us = bc_apply_vels(us, BCu, copy=False) 
        freeze(us)
        time_pred += time.time_ns() - time_pred_start

        #CORRECTOR ============================
        div_us = freeze(div_vels_to_cell(us))
        time_corr_start = time.time_ns()
        if dp_last is None:
            if   proj_variant == "non-increment":    dp_guess = pm_predicted
            elif proj_variant == "increment":        dp_guess = np.zeros_like(pm_predicted)
            elif proj_variant == "increment-rot":    dp_guess = nu*div_us
        else:
            dp_guess = dp_last
 
        freeze(dp_guess)

        # dt/us_coeff*[div(φgrad(dp))] = div(φus) - g*grad(φ)
        if proj_phase_corrector:
            div_phi_us = div_vels_to_cell([vel_phi[0]*us[0], vel_phi[1]*us[1]]) #TODO: verify when it should be div_phi_us and when just div_us
            corr_rhs = us_coeff*rho/dt*(div_phi_us - dot(wallu, grad_phi_cells))
            corr_eq = mat_div_grad((nx, ny), vel_phi)

            corr_eq = mat_apply_cell_bcs(corr_eq - mat_off(corr_rhs), BC_corr)
        # dt/us_coeff*[div(grad(dp))] = div(us)
        else:
            corr_rhs = us_coeff*rho/dt*(div_us)
            corr_eq = mat_lap((nx, ny))
            if proj_corr_p_bcs:
                corr_eq = mat_apply_cell_bcs(corr_eq - mat_off(corr_rhs), BC_corr)
            else:
                # Make sure the eq is silved with homo. neumann bcs
                # that the RHS is normallize and one DOF is pinned (othewise the matrix is singular)
                corr_rhs = corr_rhs - corr_rhs.mean()
                corr_eq = mat_apply_cell_bcs(corr_eq - mat_off(corr_rhs), BC_corr)
                mat_pin(corr_eq, nx//2, ny//2, 0, direct=False)
        A, b = mat_stencil_to_dia(corr_eq)

        global M
        if M is None or step % proj_preconditioner_every == 0 and k == 0:
            M = ilu_preconditioner(A, fill_factor=proj_preconditioner_fill_factor)
        dp, corr_iters_info = sp.sparse.linalg.bicgstab(A, -b, M=M, x0=dp_guess.ravel(), rtol=1e-5, atol=0, maxiter=proj_corr_maxiter)
        dp = dp.reshape((nx, ny))
        freeze(dp)

        dp_last = dp
        if corr_iters_info < 0: print(f"Corrector breakdown at step {step}"); return {"pred": us}
        if corr_iters_info > 0: print(f"Corrector slow convergence ({corr_iters} iters) at step {step}")

        time_corr += time.time_ns() - time_corr_start
        
        #UPDATES ============================
        grad_dp = grad_cell_to_vels(bc_expand_cell(dp, BC_corr)) 
        freeze(grad_dp)

        um = [None, None]
        um[0] = us[0] - dt/us_coeff*rho*grad_dp[0]
        um[1] = us[1] - dt/us_coeff*rho*grad_dp[1]
        um = bc_apply_vels(um, BCu, copy=False)

        if   proj_variant == "non-increment":    pm = dp
        elif proj_variant == "increment":        pm = pm_predicted + dp
        elif proj_variant == "increment-rot":    pm = pm_predicted + dp - nu*div_us

        # def norm(x): return np.sqrt(np.sum(x*x))
        # print(f"corrector L2:{norm(dp)}: sign:")
        # print(f"div pred:{norm(div_vels_to_cell(us))} div final:{norm(div_vels_to_cell(um))}")
        # print(f"lap pred:{norm(lap(pm_predicted))} lap final:{norm(lap(pm))}")

        umk = freeze(um)
        pmk = freeze(pm)

    # TODO cleanup. Move all printing to "app"!
    time_whole = time.time_ns() - time_start
    print(f"time {time_whole//1e6}ms pred {pred_iters}:{int(time_pred/time_whole*100)}% corr {time_corr//1e6}ms:{int(time_corr/time_whole*100)}%")

    # TODO: also move to "app"
    global step_total_time_sum, step_pred_time_sum, step_corr_time_sum
    step_total_time_sum += time_whole
    step_pred_time_sum += time_pred
    step_corr_time_sum += time_corr

    return {"u": umk[0], "v": umk[1], "p":pmk, "pred": us, "dp":dp}

def main():
    # TODO: test conservativness
    # TODO: implement "do nothing" boundary type
    # TODO: implement moving walls (wallu field)

    # ==============================
    #          PARAMS 
    # ==============================
    global nx, ny, nu, Lx, Ly, dx, dy, dt
    nx = 180 #num cells
    ny = 80
    nu = 1.3059e-5 #viscosity
    Ly = 1 #size of domain in meters
    Lx = Ly*nx/ny 
    dx = Lx/nx
    dy = Ly/ny
    dt = 4e-3
    t0 = 0 #begin time
    t1 = 8 #end time

    global rho, beta, phi_width, phi_eps, phi_delta, phi_cutoff
    rho = 1 #density.
    beta = 0.02 #strength of phase imposed boundary conditions
    phi_delta = 1e-6 #value we add to phi during calculations to regularize the equation in regions where phi=0 
    phi_cutoff = 1e-3 #values under/above 1 minus this are considered pure wall/pure liquid
    phi_width = 8 #width of the phase interface in cells
    phi_eps = calc_phi_eps(phi_width) #width of the phase interface (between phi_cutoff) in real units 
    # phi_width = calc_phi_w(phi_eps)

    # ==============================
    #            DOMAIN
    # ==============================
    example_fields = False
    # example_fields = True
    phase_field = True
    # phase_field = False
    phase_filed_domain = True
    # phase_filed_domain = False
    parabolic_profile = True

    # domain = {'type':"channel"}
    domain = {'type':"channel", 'circle':True, 'dot_size':0.1*Ly, 'dot_offset':2, 'dot_posx':0.2*Lx, 'dot_posy':0.5*Ly}
    # domain = {'type':"cavity"}
    # domain = {'type':"channel_cavity", "gap":0.2}
    # domain = {'type':"real_cavity"}

    # ==============================
    #      PROJECTION METHOD 
    # ==============================
    proj_outer_iters = 1 #iterations each time step to minimize splitting error caused by projection method
    # proj_variant = "non-increment"
    # proj_variant = "increment"
    proj_variant = "increment-rot"

    # Defines the BDFq order of the time stepping. 
    # Allowed values are {1, 2, 3}
    proj_bdf_order = 1

    # Which order of extrapolation to use for the predicated values in NSE
    # predictor. Allows {1, 2, 3} 
    # 1 = previous value (first order)
    # 2 = linear extrapolation (second order)
    # 3 = quadratic extrapolation (third order)
    # These should be equal to proj_bdf_order unless there is some problem with convergence
    proj_extrapolate_u = 2 
    proj_extrapolate_p = 2 

    # Whether to use previous iteration/extrapolated 
    # or previous non-linear iterate as 
    # the other velocity in advection. 
    proj_nonlinear_predictor = True 
    
    # Whether to use the proper phase-avare poisson solve or 
    # the classic phase-oblivious poisson solve
    proj_phase_corrector = False
    proj_corr_p_bcs = True

    # preconditioner applied to pressure solve
    proj_preconditioner_every = 30
    proj_preconditioner_fill_factor = 35
    
    proj_pred_maxiter = 100
    proj_corr_maxiter = 1000
    
    proj_phase_cutoff = 0.1 #applied during iterative solve of velocity

    advect_params = AdvectParams() 
    advect_params.limiter = "vanalbada"
    advect_params.variant = "advect"
    advect_params.blend_factor = 0.85

    # ==============================
    #      VISUALISATION 
    # ==============================
    display_pause = 0 #pause in seconds after each iteration for debugging
    display_every = 25 #update display every X iters. (matplotlib is slow)

    # display_field = ""
    # display_field = "p"
    # display_field = "gradp"
    # display_field = "u"
    # display_field = "v"
    display_field = "velmag"
    # display_field = "predu"
    # display_field = "predv"
    # display_field = "predmag"
    # display_field = "divpred"
    # display_field = "dp"
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
    display_phase_cutoff = True
    display_velocity_arrows = False
    display_face_velocity_arrows = False
    display_face_velocity_arrows_offsets = False
    display_streamlines = False 

    # ==============================
    #      INITIAL CONDITIONS 
    # ==============================
    boundaries, sdf = make_domain(domain, 0, parabolic_profile, phase_filed_domain)
    phi = sdf_to_phase_field(sdf)
    phi_outline = marching_squares(sdf, 0)

    cutoff_outline = marching_squares(phi, proj_phase_cutoff)
    
    fields = {
        "u": np.zeros((nx+1, ny)),
        "v": np.zeros((nx, ny+1)),
        "p": np.zeros((nx, ny)),
        "f": phi,
        "sdf": sdf,
    }
    past_fields = []
    max_history_len = max(proj_bdf_order, proj_extrapolate_u, proj_extrapolate_p, 1)

    
    # ==============================
    #      MAIN LOOP
    # ==============================
    plt.ion()
    fig = plt.figure(figsize=(6*Lx/Ly, 6), dpi=100)
    step = -1
    t = t0 - dt

    start = time.time_ns()

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
        # u_in = 1
        boundaries, sdf = make_domain(domain, u_in, parabolic_profile, phase_filed_domain)
        low_bounds = Boundary.to_low_bounds(boundaries.values(), dictify=False)

        past_fields = [fields] + past_fields
        while len(past_fields) > max_history_len:
            past_fields.pop()

        # SIMULATE ===========================
        if example_fields:
            new_fields = generate_example_fields()
        else:
            assert phase_field #TODO build nnon phase field variant
            new_fields = step_phase(
                fields, low_bounds,
                step=step,
                past_fields = past_fields,
                proj_preconditioner_every = proj_preconditioner_every,
                proj_preconditioner_fill_factor = proj_preconditioner_fill_factor,

                proj_phase_cutoff = proj_phase_cutoff,
                proj_pred_maxiter = proj_pred_maxiter,
                proj_corr_maxiter = proj_corr_maxiter,

                proj_bdf_order = proj_bdf_order,
                proj_extrapolate_u = proj_extrapolate_u, 
                proj_extrapolate_p = proj_extrapolate_p, 

                proj_phase_corrector=proj_phase_corrector,
                proj_corr_p_bcs = proj_corr_p_bcs,
                proj_nonlinear_predictor = proj_nonlinear_predictor,
                proj_variant=proj_variant, 
                proj_outer_iters=proj_outer_iters, 
                advect_params=advect_params)

        fields.update(new_fields)
        #PLOTTING ============================
        if step % display_every == 0:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_title(f"step = {step} t = {float(t):.6}")
            plot(fig, ax, fields, boundaries, low_bounds,
                display_field=display_field,
                phi_outline = phi_outline,
                cutoff_outline = cutoff_outline,
                line_color = line_color,
                display_phase_walls = display_phase_walls,
                display_phase_cutoff = display_phase_cutoff,
                display_cell_centers = display_cell_centers,
                display_grid = display_grid,
                display_BCs = display_BCs,
                display_velocity_arrows = display_velocity_arrows,
                display_face_velocity_arrows = display_face_velocity_arrows,
                display_face_velocity_arrows_offsets = display_face_velocity_arrows_offsets,
                display_streamlines = display_streamlines,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            if display_pause > 0:
                time.sleep(display_pause) 

    dur = time.time_ns() - start

    global step_total_time_sum, step_pred_time_sum, step_corr_time_sum

    print(f"took {dur*1e-9}s")
    print(f"step:{step_total_time_sum*1e-9}s pred:{step_pred_time_sum*1e-9}s ({int(step_pred_time_sum/step_total_time_sum*100)}%) corr:{step_corr_time_sum*1e-9}s ({int(step_corr_time_sum/step_total_time_sum*100)}%)" )
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

def make_domain(domain:dict, u_in:float, parabolic:bool, phase_field:bool) -> Tuple[Boundaries, np.ndarray]:
    def inflow_profile(u:float, n:int) -> np.ndarray:
        if parabolic == False:
            return np.full(n, u)
        centers = (np.arange(n) + 0.5)/n
        profile = u*(1 - (2*centers - 1)**2)
        return profile

    sdf = np.full((nx, ny), 1000)
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

        wall = np.zeros((nx, ny), dtype=bool)
        wall[:, :w] = 1
        wall[:, -w:] = 1
        sdf = sdf_from_mask(wall)

        if 'circle' in domain:
            r   = domain['dot_size']
            px  = domain['dot_posx']
            py  = domain['dot_posy']
            off = domain['dot_offset']
            
            x_centers = (np.arange(nx) + 0.5) * dx
            y_centers = (np.arange(ny) + 0.5) * dy
            Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')
            dot_sdf = np.hypot(Xc - px, Yc - off*dy - py) - r
            sdf = np.minimum(sdf, dot_sdf)

    elif domain_variant == "channel_cavity" and phase_field == True:
        w = phi_width//2 #wall width
        h = max((ny - 2*w)//5, 1)
        inflow = inflow_profile(u_in, h)

        wall = np.zeros((nx, ny), dtype=bool)
        wall[:w, :] = 1
        wall[-w:, :] = 1
        wall[:, :w] = 1
        wall[:, -w:] = 1
        wall[:w, ny-(h+w):ny-w] = 0
        wall[-w:, w:h+w] = 0
        sdf = sdf_from_mask(wall)
        
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
    else:
        raise ValueError(f"Unknown domain variant / phase field {domain_variant=} {phase_field=}")

    return (boundaries, sdf)

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
    return 0.5*(1 - np.tanh(-3*sdf/phi_eps))

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
    cutoff_outline = None,
    line_color = "white",
    display_phase_walls = False,
    display_phase_cutoff = False,
    display_cell_centers = False,
    display_grid = False,
    display_BCs = True,
    display_velocity_arrows = False,
    display_face_velocity_arrows = True,
    display_face_velocity_arrows_offsets = False,
    display_streamlines = False,
):
    dd = min(dx, dy)

    ex_p = bc_expand_cell(fields["p"], low_bounds['p'])
    uex = bc_expand_velx(fields["u"], low_bounds['u'])
    vex = bc_expand_vely(fields["v"], low_bounds['v'])

    ax.set_xlim(-dx, (nx + 1) * dx)
    ax.set_ylim(-dy, (ny + 1) * dy)
    ax.set_aspect('equal')

    velx = interp_vels_to_cellx(uex)
    vely = interp_vels_to_celly(vex)
    velmag = np.hypot(velx, vely)

    # Field drawing
    display_field_tuple = None
    if   display_field == "p":      display_field_tuple = (ex_p, "pressure")
    elif display_field == "u":      display_field_tuple = (velx, "velocity u")
    elif display_field == "v":      display_field_tuple = (vely, "velocity v")
    elif display_field == "velmag": display_field_tuple = (velmag, "velocity magnitude")
    
    elif display_field == "gradp":      
        grad_p = grad(ex_p)
        mag = np.hypot(grad_p[0], grad_p[1])
        display_field_tuple = (bc_expand_cell(mag, low_bounds['p']), "gradp")

    elif display_field in ["predu", "predv", "predmag"] and "pred" in fields:
        epred = bc_expand_vels(fields["pred"], (low_bounds['u'], low_bounds['v']))
        predu = interp_vels_to_cellx(epred[0]) 
        predv = interp_vels_to_celly(epred[1])
        if display_field == "predu": display_field_tuple = (predu, "predictor v")
        if display_field == "predv": display_field_tuple = (predv, "predictor u")
        if display_field == "predmag": display_field_tuple = (np.hypot(predu, predv), "predictor magnitude")
    elif display_field in ["psiu", "psiv", "psimag"]:
        ru = (uex[:-2,1:-1] - uex[1:-1,1:-1])/(uex[1:-1,1:-1] - uex[2:,1:-1])
        rv = (vex[1:-1,:-2] - vex[1:-1,1:-1])/(vex[1:-1,1:-1] - vex[1:-1,2:])

        psiu = np.zeros_like(uex)
        psiu[1:-1, 1:-1] = np.minimum((ru + np.abs(ru)) / (1 + np.abs(ru)), 1)
        
        psiv = np.zeros_like(vex)
        psiv[1:-1, 1:-1] = np.minimum((rv + np.abs(rv)) / (1 + np.abs(rv)), 1)
        
        ipsi = [interp_vels_to_cellx(psiu), interp_vels_to_celly(psiv)]
        if display_field == "psiu": display_field_tuple = (psiu[:-1,:], "psiu")
        if display_field == "psiv": display_field_tuple = (psiv[:,:-1], "psiv")
        if display_field == "psimag": display_field_tuple = (np.hypot(ipsi[0], ipsi[1]), "psimag")
    elif display_field == "dp" and "dp" in fields:
        display_field_tuple = (bc_expand_cell(fields["dp"], low_bounds['p']), "pressure correction (dp)")
    elif display_field == "phase" and "phase" in fields:
        display_field_tuple = (bc_expand_cell(fields["f"], low_bounds['f']), "phase")
    elif display_field == "sdf" and "sdf" in fields:
        display_field_tuple = (bc_expand_cell(fields["sdf"], low_bounds['f']), "sdf")

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
        zoomed = [zoom(s, zoom=scale, order=1) for s in stacked] if scale < 1 else stacked
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
        ax.streamplot(Xc[:,0], Yc[0,:], velx[1:-1,1:-1].T, vely[1:-1,1:-1].T, color=line_color, density=density, linewidth=lw, arrowsize=0.7)

    if display_phase_walls and phi_outline is not None and len(phi_outline) > 0:
        dx_dy = np.array([dx, dy])
        e1 = (phi_outline[:, 0:2] + 0.5) * dx_dy
        e2 = (phi_outline[:, 2:4] + 0.5) * dx_dy
        lc = LineCollection(np.stack([e1, e2], axis=1), colors=line_color, linewidths=1, capstyle="butt")
        ax.add_collection(lc)
    
    if display_phase_cutoff and cutoff_outline is not None and len(cutoff_outline) > 0:
        dx_dy = np.array([dx, dy])
        e1 = (cutoff_outline[:, 0:2] + 0.5) * dx_dy
        e2 = (cutoff_outline[:, 2:4] + 0.5) * dx_dy
        lc = LineCollection(np.stack([e1, e2], axis=1), colors="red", linewidths=1, capstyle="butt")
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