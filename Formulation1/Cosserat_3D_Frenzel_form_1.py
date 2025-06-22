from dolfinx import mesh, fem, io, plot
from dolfinx.common import Timer
from dolfinx.fem.petsc import LinearProblem
from ufl import (Identity, Measure, PermutationSymbol,
                nabla_grad, tr, TrialFunctions, TestFunctions,  
                dot, inner, sym, dx, skew)
from basix.ufl import element, mixed_element
from petsc4py.PETSc import ScalarType   
from mpi4py import MPI
import numpy as np
import pyvista as pv

pv.global_theme.transparent_background = True

### Parameters
# units are in N, tonn, mm, s, mJ
Nx = 5 # Ny = Nx = Nz/3, number of cells in x direction
Nz = 15
a_x = 1.5e-4  # cell costant for x,y
a_z = 2/3 * a_x  # cell costant for z
perc_displ = 0.015
nu = -0.125 # Poisson's ratio
G = 0.1 # shear modulus G
N = 0.93 # coupling parameter
l = 6*a_x  # intrinsic length scale


λ = G*((2*nu)/(1-2*nu))
μ =  G*((1-2*N**2)/(1-N**2))
# Computing COSSERAT constants
γ = 4*G*l**2
κ = G*((2*N**2)/(1-N**2))
# Micropolar elastic constants
α = ( μ * N**2 ) / (N**2 -1.0)
β = μ * l

E_3 = PermutationSymbol(3)  # Levi-Civita tensor for n = 3
I_3 = Identity(3)  # Identity tensor for n = 3
ϵ = lambda u, theta : nabla_grad(u) - dot(E_3,theta)  # micropolar strain
ϵ_sym = lambda ϵ : sym(ϵ)
ϵ_skew = lambda ϵ: skew(ϵ) 

ϕ = lambda theta : nabla_grad(theta)  # micro-curvature
### Stresses
def σ_B(ϵ):
    """Evaluating Boltzmann part of stress tensor sigma"""
    e_sym = ϵ_sym(ϵ)
    stress = λ*tr(e_sym)*I_3+ (2*μ + κ)*e_sym
    return stress

σ_C = lambda ϵ: κ*ϵ_skew(ϵ)  # micro-continuum coupling
σ   = lambda ϵ: σ_B(ϵ) + σ_C(ϵ)

m_R = lambda θ : γ *ϕ(θ)  # couple stress

msh  = mesh.create_box(
comm=MPI.COMM_WORLD, 
points=([np.array([0, 0,0]),np.array([L:=Nx*a_x, L,  H:=Nz*a_z])]),
n= (5,5,20),
cell_type= mesh.CellType.hexahedron         
)

# displaying box mesh
cells, types, x = plot.vtk_mesh(msh)
grid = pv.UnstructuredGrid(cells, types, x)
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.show()

# Looking for top (z=H) and bottom (z=0) facet of the box x[2]=z
boundaries = [(1, lambda x: np.isclose(x[2], 0)),
            (2, lambda x: np.isclose(x[2],H))]

facet_indices, facet_markers = [], []
fdim = msh.topology.dim - 1
# sort and tag facet to create a subdomain for creating a ds Measure
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(msh, fdim, locator)
    facet_indices = np.append(facet_indices, facets)
    facet_markers = np.append(facet_markers, np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


# Function Space
dim = msh.geometry.dim
V_el = element("Lagrange", msh.topology.cell_name(), 1, shape=(dim,)) # displacement space
W_el = element("Lagrange", msh.topology.cell_name(), 2, shape=(dim,)) # micro rotation space
VW_element = mixed_element([V_el, W_el]) # dim 6
V = fem.functionspace(msh, VW_element) # mixed function space
# Extracting subspaces for displacement and micro-rotation    
V0,_ = V.sub(0).collapse()
V1,_ = V.sub(1).collapse() 

u, θ = TrialFunctions(V)
v, η = TestFunctions(V)

### Boundary Conditions
facets_dim = msh.topology.dim - 1

boundary_facets = mesh.locate_entities_boundary(
    msh = msh,
    dim= facets_dim, 
    marker=lambda x: np.isclose(x[2], 0.0) 
    )

boundary_facets_z = mesh.locate_entities_boundary(
    msh = msh,
    dim= facets_dim, 
    marker=lambda x: np.isclose(x[2], H) 
    )


dofs = fem.locate_dofs_topological(V=(V.sub(0), V0),entity_dim=facets_dim,entities=boundary_facets)
dofs_rot = fem.locate_dofs_topological(V=(V.sub(1), V1),entity_dim=facets_dim,entities=boundary_facets)
dofs_z = fem.locate_dofs_topological(V=(V.sub(0), V0),entity_dim=facets_dim,entities=boundary_facets_z) 
dofs_z_rot = fem.locate_dofs_topological(V=(V.sub(1), V1),entity_dim=facets_dim,entities=boundary_facets_z)

# interpolation of the boundary conditions for the displacement 
fixed_displ_expr= lambda x: np.zeros((3, x.shape[1]), dtype=ScalarType)
fixed_displ = fem.Function(V0)
fixed_displ.interpolate(fixed_displ_expr)
# interpolation of the boundary conditions for the micro rotation
fixed_rot_expr= lambda x: np.zeros((3, x.shape[1]), dtype=ScalarType)
fixed_rot = fem.Function(V1)
fixed_rot.interpolate(fixed_rot_expr)


# fixing displacement along the z axis for z=H, instead of applying a forcing term
def z_displ_expr(x):
    values = np.zeros((3, x.shape[1]), dtype=np.float64)
    values[2] = -perc_displ*H
    return values

fixed_displ_z = fem.Function(V0)
fixed_displ_z.interpolate(z_displ_expr)

def z_rot_expr(x):
    values = np.zeros((3, x.shape[1]), dtype=np.float64)
    return values

fixed_rot_z = fem.Function(V1)
fixed_rot_z.interpolate(z_rot_expr)

bc_displ = fem.dirichletbc(value=fixed_displ,dofs=dofs,V=V.sub(0))
bc_displ_z = fem.dirichletbc(value=fixed_displ_z,dofs=dofs_z,V=V.sub(0)) 
bc_rot = fem.dirichletbc(value=fixed_rot,dofs=dofs_rot,V=V.sub(1))
# If you want to fix rotation on the top facet add this to bcs list
bc_rot_z = fem.dirichletbc(value=fixed_rot_z,dofs=dofs_z_rot,V=V.sub(1))

bcs= [bc_displ, bc_rot, bc_displ_z] 


ds = Measure("ds", domain=msh, subdomain_data=facet_tag)

### Variational Formulation 2


# Defining weak form -----------------------------------
forcing_components = ScalarType((0.0, 0.0, 0.0))
traction_components = ScalarType((0.0, 0.0, 0.0))

forcing = fem.Constant(msh,forcing_components)
t = fem.Constant(msh, traction_components)
# if the forcing of traction term is not null, the displacement along the z axis is not fixed and removed\\
# and removed from the list of boundary conditions
if (forcing_components != ScalarType((0.0, 0.0, 0.0))).any():
    bcs.remove(bc_displ_z)
if (traction_components != ScalarType((0.0, 0.0, 0.0))).any():
    bcs.remove(bc_displ_z)

ϵ_u = ϵ(u, θ)
ϵ_v = ϵ(v, η)
σ_u = σ(ϵ_u) 
χ_u = ϕ(θ)

ds = Measure("ds", domain=msh, subdomain_data=facet_tag)
### left and right hand side of the weak form formulation--------------
a = (
    inner(σ_u, ϵ_v) * dx +
    inner(ϕ(η),m_R(θ)) * dx - 
    inner(dot(η,E_3), σ_C(ϵ_u)) * dx
    )
L = inner(t, v) *ds(2) + inner(forcing, v) * dx
### Solving the problem


### Compute solution --------------------------------------------------- 
problem = LinearProblem(
                        a=a, 
                        L = L,
                        bcs= bcs,
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
                        )
with Timer() as t:
    uh = problem.solve()
    print(f"Elapsed time: {t.elapsed()}")
u_h, θ_h = uh.sub(0).collapse(), uh.sub(1).collapse()
u_h.name= 'displacement'
θ_h.name = 'micro_rotation'

### Displaying solutions

p = pv.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V0)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = u_h.x.array.reshape(geometry.shape[0], 3)
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pv.OFF_SCREEN:
    p.show()

p = pv.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V1)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid["theta"] = θ_h.x.array.reshape(geometry.shape[0], 3)
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("theta", factor=1)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pv.OFF_SCREEN:
    p.show()
    
# function to store both displacement and micro-rotation
total_displ = fem.Function(V1)
space = V1  # FunctionSpace for plotting id V_el.degree >= W_el.degree
if V_el.degree == W_el.degree:
    U_values = u_h.x.array + θ_h.x.array  # adding values 
elif V_el.degree > W_el.degree:
    # interpolating the higher degree element over the lower one
    displ_expr = fem.Expression(u_h, V1.element.interpolation_points())
    displ = fem.Function(V1)
    displ.interpolate(displ_expr)
    U_values = displ.x.array + θ_h.x.array
else:
    space = V0  # changing the function space if V_el < W_el
    # interpolating the higher degree element over the lower one
    total_displ = fem.Function(V0)
    rot_expr = fem.Expression(θ_h, V0.element.interpolation_points())
    rot = fem.Function(V0)
    rot.interpolate(rot_expr)
    U_values = u_h.x.array + rot.x.array

total_displ.name = 'combined_displacement'
total_displ.x.array[:] = U_values  # assigning values to the function

# Displayng displacement and rotations togheter
p = pv.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(space)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid["U"] = U_values.reshape(geometry.shape[0], 3)
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("U", factor=1)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pv.OFF_SCREEN:
    p.show() 

### saving solution to a .pvd file
with io.VTKFile(MPI.COMM_WORLD, __file__.rstrip('.py') + "/displacement_Nx" + str(Nx) + ".pvd", "w") as vtk:
    vtk.write_function(u_h, 0.0)  # 0.0 = time step

with io.VTKFile(MPI.COMM_WORLD, __file__.rstrip('.py') + "/micro_rotation_Nx" + str(Nx) + ".pvd", "w") as vtk:
    vtk.write_function(θ_h, 0.0)  # 0.0 = time step

with io.VTKFile(MPI.COMM_WORLD, __file__.rstrip('.py') + "/u_Nx" + str(Nx) + ".pvd", "w") as vtk:
    vtk.write_function(total_displ, 0.0) # 0.0 = time step
