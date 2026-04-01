# Literature review

## Broad topic: 
PISO/projection algorithm on collocated grids, 
projection as approximations to monolithic matrix forumlation

## Individual files

### COMPUTATIONAL FLUID DYNAMICS by Jiri Blazek Book 3rd ed
more of a literature review, very technical, not very useful for me

### An Introduction to Computational Fluid Dynamics by H K Versteeg and W Malalasekera 2nd ed
very useful book focused on the engineering side of it, SIMPLE algorithm etc. Only discusses ordered grids and is a bit outdated cause of it but its fine since I also only deal with such grids.Has worked examples that go through the entire process from discretization, BCs to forming the matrix eqs to running the methods. Super useful.
### R.I. Issa PISO algorithm: https://doi.org/10.1016/0021-9991(86)90100-2
the original paper. I had an issue with trying to implement it on collocated grids and reasonably so as the original algo is for staggered grids. 

### Stable fluids https://dl.acm.org/doi/pdf/10.1145/311535.311548
mainly for comparison

### Stable, Circulation-Preserving, Simplicial Fluids
interesting continuation on stable fluids, havent looked into it

### Numerical Solution of the Navier-Stokes Equations (projection method) by Chorin
just for citation, very very old

### A nuerical method for Solving incompressibe Vsicous flow problems by Chorin
also just for citation

### Development of a fast fluid dynamics model based on PISO algorithm for simulating indoor airflow
somehow another stable fluids??

### Implicitly coupled finite volume algorithms by Uroić, Tessa Doctoral thesis / Disertacija
someones review of the different algos

### Notes on CFD https://doc.cfd.direct/notes/cfd-general-principles/recommended-boundary-condition
practical notes from the authors of openfoam

### An overview of projection methods for incompressible flows .L. Guermond https://www.math.purdue.edu/~shen7/pub/remarks_revised.pdf
very useful overview especially around solvability with concrete boundary conditions. Also discusses incremental forms and their advantages

### A comparative study of finite volume pressure-correction projection methods on co-located grid arrangements by R. Abbasi
which of the projection method modifications work and which dont on collocated grids

### Iterated pressure-correction projection methods for the unsteady incompressible Navier–Stokes equations
how does iteration relate to the exact solution of the collocated system. Very useful.

### An efficient method for the incompressible Navier–Stokes equations on irregular domains with no-slip boundary conditions, high order up to the boundary
I am unsure if I read this

### Projection method 1: convergence and numerical boundary layers https://www.researchgate.net/profile/Jian-Guo-Liu-4/publication/243096354_Projection_Method_I_Convergence_and_Numerical_Boundary_Layers/links/540dacf30cf2f2b29a39cecf/Projection-Method-I-Convergence-and-Numerical-Boundary-Layers.pdf
Very mathematical series of pretty old papers. Not very useful
better link https://web.math.princeton.edu/~weinan/papers/cfd2

### Simple iteration method wiki https://encyclopediaofmath.org/wiki/Simple-iteration_method
Idk

### A numerical investigation of explicit pressure-correction projection methods for incompressible flows https://www.tandfonline.com/doi/epdf/10.1080/19942060.2015.1004810?needAccess=true
Maybe good for bechmarks a lot of tables

### A Review of Splitting Errors for Approximate Projection Methods  https://arc.aiaa.org/doi/epdf/10.2514/6.2003-4236
Not sure if I read it

### PERFORMANCE OF PROJECTION METHODS FOR LOW-REYNOLDS-NUMBER FLOWS Fabricio S. Sousa https://congress.cimne.com/iacm-eccomas2014/admin/files/filePaper/p2694.pdf
Another core paper discussing the approximation of projection methods. Has nice concise block matrix notation. Very useful

### Approximate projection methods and time integration stability C.D. Moen https://flair.monash.edu/intranet/proceedings/3mit/data/content/0769/paper.pdf
Pretty strange paper I didnt fully understand about possible different values of some constant during approximation within projection method. Not very useful.

### Quantifying the checkerboard problem to reduce numerical dissipation https://arxiv.org/pdf/2408.06821v1 
Very useful paper. Need to reread this

### Implementation of the Divergence-Free and Pressure-Oscillation-Free Projection Method for Solving the Incompressible Navier-Stokes Equations on the Collocated Grids by Yu-Xin Ren
Paper that blames incosistent discretization for the pressure field oscilations. That is the laplacian during poisson solve should be on grid with 2h instead of grid of h. It recommends solving poisson on on h and then solving another poisson to filter the solution onto 2h grid. Tried it and it didnt work for me.

### A Second-Order Accurate Pressure-Correction Scheme for Viscous Incompressible Flow by J. van Kan
I dont think I have read this as I dont have access to it.

### On incremental projection methods by L. GUERMOND and L. QUARTAPELLE
Investigation of splitting error both mathematically and practically. Not very important.

### DISCUSSION ON MOMENTUM INTERPOLATION METHOD FOR COLLOCATED GRIDS OF INCOMPRESSIBLE FLOW by Bo Yu et al.
https://nht.xjtu.edu.cn/paper/en/2002206.pdf
Some momentum interpolation technique ala QUICK. Not very important

### Computational fluid-structure interaction with the moving immersed boundary method by Shang-Gui CAI 
https://theses.hal.science/tel-01461619v1/file/These_UTC_Shang_Gui_Cai.pdf
Very good overview of the different methods and implementation and benchamrks. Definitely go over this again to validate everything. Read again!

### A Second-Order Accurate Pressure-Correction Scheme for Viscous Incompressible Flow by Jos J van Kan
https://www.researchgate.net/publication/243771948_A_Second-Order_Accurate_Pressure-Correction_Scheme_for_Viscous_Incompressible_Flow
I believe the first staggered grid impl. Very importat. Cite and use as validation. Read again!

## On the design
I initially wanted to do collocated grid cause it seemed that the single grid would be easier to extend to the phase field/wall intersections just because its simpler. That however posed problems in all of the methods due to checkerboarding even on 1D grids. The only known way to circumwent this is to use Rhye-Chow interpolation. That however needs access to the symbolic digonal matrix coeefficient. This poses an implementation problem: implementing the methods by explicit matrix construction is very difficult due to the irregularity of BCs and upwind (+ in the future with the phase field). It turns out that without it we can use GC-like methods that never need the explicit matrix and only ever need the application of the matrix onto a vector as function. This is very very handy as it means we can do "just a explicit calculation*" and pass it into the matrix solver - with the * that the calculation needs to be linear (duh) and needs to handle the constant offsets in a special way. The boundary conditions create additional source terms that should be moved to the RHS. We can either do that explicitly which once again requires some implicit treatment OR use a trick of simply adding them up to everything else and then at the end correcting them. To correct we simply apply the calculation onto 0 vector and then substract from both sides of the EQ. With this the staggered grid impl is super simple and straight forward.

While doing this I realized that we can use similar procedure to extract the diagonal coefficients needed for Rhye-Chow. Just supply 1-0-1-0... alternating vector into the procedure OR a red-black style checkerboard depedning on the stencils used and then the procedure will evaluate one half of the diagonal entries. The suply 0-1-0-1... vector to get the other half! This extends to more complicated stencils but is less practical

## Problems and interesting findings

### Boundary conditions
The boundary conditions usually given on inlet of  dirichlet veclotiy zero neumann pressure are not entirely correct. (In contrast, outlet and walls are fine). However projection method requires this to be the case else it explodes. This causese slight errors in the pressure field that spread into oscilations when unchecked. The projection method incremental and rotational form should help correct exactly this boundary condition induced errors yet because it only does so using the error from the correct, when near convergence it doesnt do anything and the errors only amplify with time. In contrast the non-incremental form resets pressure every step so that the errors are tiny and dont grow.

