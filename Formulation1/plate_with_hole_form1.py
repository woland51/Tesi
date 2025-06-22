from dolfinx import mesh, fem,plot, io
from dolfinx.common import Timer
from dolfinx.fem.petsc import LinearProblem
from ufl import (Identity, Measure, PermutationSymbol, MixedFunctionSpace,
                grad, tr, TrialFunctions, TestFunctions, split, skew,  
                dot, inner, sym, sqrt, dx)
from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import pyvista as pv
import numpy as np
import logging
from mesh_utils import plate_with_hole_field


pv.global_theme.transparent_background = True  #setting backgroun for pyvista screenshot
logging.basicConfig(format='%(levelname)-8s : %(message)s',
                    level=logging.INFO)

# units are in N, tonn, mm, s, mJ
scale = 1.0e-3 # scale factor for the units, 1e-3 for [m], 1 for [mm]
G  = ((1/scale)**2)*50e3    # Shear modulus [MPa]
ν = 0.3  # Poisson's ratio
l  = scale * 30  # Characteristic length (bending) [mm]
N  = 0.9  # coupling number
trac = ((1/scale)**2)*1.0e-3 # traction [MPa]

N_values = np.linspace(0.1, 0.9, 9)  # values for N simulation
maxLengthElement = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # values for mesh size simulation

def main(minElement=1e-3, maxElement=1e-2, N = 0.9, display = False, save = False) -> float:
    img_dir = "Immagini"  # directory to store pyvista screenshot

    λ= G*((2*ν)/(1-2*ν))
    μ=  G*((1-2*N**2)/(1-N**2))
    # Computing COSSERAT constants
    γ = 4*G*l**2
    κ = G*((2*N**2)/(1-N**2))
    # create the mesh
    msh,_, facet_tags = plate_with_hole_field(maxLength=maxElement,
                                                    minLength=minElement,
                                                    save=False,)

    # displaying plate with hole mesh
    if display:
        cells, types, x = plot.vtk_mesh(msh)
        grid = pv.UnstructuredGrid(cells, types, x)
        p = pv.Plotter()
        p.add_mesh(grid, show_edges=True)
        # p.set_background("white")
        p.camera.tight(0.1, view='xy')
        p.show_axes()
        p.show(auto_close=False, interactive=False)
        p.screenshot("".join([img_dir,"/mesh_xy_view.png"]))
        p.close()

    dim = msh.geometry.dim
    # Definig element and mixed element function space
    V_el = element("Lagrange", msh.topology.cell_name(), 2, shape=(dim,))  # element for displacement
    W_el = element("Lagrange", msh.topology.cell_name(), 1,)  # element for micro-rotation
    VW_el = mixed_element([V_el, W_el])
    V = fem.functionspace(msh, VW_el)
    # Extracting subspaces for displacement and micro-rotation
    V0, V0_dofs = V.sub(0).collapse()
    V1, V1_dofs = V.sub(1).collapse()
    # Trial and Test functions
    (u_trial, theta_trial) = TrialFunctions(V) 
    (v_test, zeta_test) = TestFunctions(V)

    fdim = msh.topology.dim - 1
    # Finding bottom and left boundary facet using facet_tags defined in gmsh
    boundary_left = facet_tags.find(1)
    boundary_bottom =facet_tags.find(4)
    # Looking for dofs for each boundary
    dofs_left_x = fem.locate_dofs_topological(V=(V.sub(0).sub(0), V0),entity_dim=fdim,entities=boundary_left)
    dofs_bottom_y = fem.locate_dofs_topological(V=(V.sub(0).sub(1), V0),entity_dim=fdim,entities=boundary_bottom)

    # dolfinx function to express bc values, in mixed spaces every bc value
    # must be provided as a Function
    u0 = fem.Function(V0)
    u0.interpolate(lambda x: np.zeros((2, x.shape[1])))

    bc_left_x = fem.dirichletbc(u0,dofs=dofs_left_x,V=V.sub(0).sub(0))  # Imposing x = 0 (x[0]) on the left
    bc_bottom_y = fem.dirichletbc(u0,dofs=dofs_bottom_y,V=V.sub(0).sub(1))  # Imposing y = 0 (x[1]) on the bottom

    dofs_theta_left = fem.locate_dofs_topological(V=(V.sub(1), V1),entity_dim=fdim,entities=boundary_left)
    dofs_theta_bottom = fem.locate_dofs_topological(V=(V.sub(1), V1),entity_dim=fdim,entities=boundary_bottom)

    theta0 = fem.Function(V1)
    theta0.interpolate(lambda x: np.zeros((1, x.shape[1])))
    # Imposing theta = 0 for both boundaries
    bc_theta_left = fem.dirichletbc(theta0,dofs=dofs_theta_left,V=V.sub(1))
    bc_theta_bottom = fem.dirichletbc(theta0,dofs=dofs_theta_bottom,V=V.sub(1))  

    bcs = [bc_left_x, bc_bottom_y, bc_theta_left, bc_theta_bottom]  # Collectin bcs in a single list

    ### Defining micropolar strain and stresses

    # Micropolar strain:
    E_2 = PermutationSymbol(2)  # Levi-Civita tensor for n = 2
    I_2 = Identity(2)  # Identity tensor for n = 2
    ϵ = lambda u, theta : grad(u) - E_2*theta  # micropolar strain (non symmetric)
    ϵ_sym = lambda ϵ : sym(ϵ)  # symmetric part of strain tensor
    ϵ_skew = lambda ϵ: skew(ϵ) # anti symmetric part of strain tensor
    ϕ = lambda theta : grad(theta)  # micro-curvature

    # Stresses
    def σ_B(ϵ):
        """Evaluating Boltzmann part of stress tensor sigma"""
        e_sym = ϵ_sym(ϵ)
        stress = λ*tr(e_sym)*I_2 + (2*μ + κ)*e_sym
        return stress

    σ_C = lambda ϵ: κ*ϵ_skew(ϵ)  # micro-continuum coupling
    σ   = lambda ϵ: σ_B(ϵ) + σ_C(ϵ)
    m_R = lambda θ : γ *ϕ(θ)  # couple stress

    # Defining weak form -----------------------------------
    forcing = fem.Constant(msh, ScalarType((0.0, 0.0)))
    traction = fem.Constant(msh, ScalarType((trac, 0.0)))
    # Evaluating epsilon and sigma for test and trial functions
    ϵ_u = ϵ(u_trial, theta_trial)
    ϵ_v = ϵ(v_test, zeta_test)
    σ_u = σ(ϵ_u) 
    # Definin a measure over the subdomain of the mesh 1=left, 2=top, 3=right, 4=bottom
    ds = Measure("ds", domain=msh, subdomain_data=facet_tags)
    ### left and right hand side of the weak form formulation--------------
    # Bilinear form
    a = (
        inner(σ_u, ϵ_v) * dx +
        inner(ϕ(zeta_test),m_R(theta_trial)) * dx - 
        inner(zeta_test*E_2, σ_C(ϵ_u)) * dx
        )
    # Linear form
    L = inner(traction, v_test) *ds(3) + inner(forcing, v_test) * dx
    ### Solving the problem --------------------------------------------------- 
    # Assembling the linear problem
    problem = LinearProblem(
                            a=a, 
                            L = L,
                            bcs= bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
                            )
    # Compute solution 
    with Timer() as t:
        uh = problem.solve()
        print(f"Elapsed time: {t.elapsed()}")
    u, θ = uh.sub(0).collapse(), uh.sub(1).collapse()
    
    ### Compute von Mises stress
    ϵ_uh = ϵ(u, θ)  # micropolar strain 
    σ_uh = σ(ϵ_uh)  # stress tensor

    deviatoric_stress_tensor = (σ_uh - 1/3 * tr(σ_uh) * I_2)
    von_Mises_stress = sqrt(3/2 * inner(deviatoric_stress_tensor, deviatoric_stress_tensor))
 
    # Interpolating von Mises stress over a 0 degree Discontinuos Lagrange element for visualization
    V_von_mises = fem.functionspace(msh, ("DG", 0))
    stress_expr = fem.Expression(von_Mises_stress, V_von_mises.element.interpolation_points())
    stresses = fem.Function(V_von_mises)
    stresses.interpolate(stress_expr)

    # Extracting components of stress tensor
    σxx = σ_uh[0, 0] 
    σyy = σ_uh[1, 1] 
    σxy = σ_uh[0, 1] 
    σyx = σ_uh[1, 0] 
    # Interpolating each component over a DG 0 space
    σ_function_space = fem.functionspace(msh, ("DG", 0))
    σxx_expr = fem.Expression(σxx, σ_function_space.element.interpolation_points())
    σxx = fem.Function(σ_function_space)
    σxx.interpolate(σxx_expr)

    σyy_expr = fem.Expression(σyy, σ_function_space.element.interpolation_points())
    σyy = fem.Function(σ_function_space)
    σyy.interpolate(σyy_expr)

    σxy_expr = fem.Expression(σxy, σ_function_space.element.interpolation_points())
    σxy = fem.Function(σ_function_space)
    σxy.interpolate(σxy_expr)

    σyx_expr = fem.Expression(σyx, σ_function_space.element.interpolation_points())
    σyx = fem.Function(σ_function_space)
    σyx.interpolate(σyx_expr)

    SCF=max(σxx.x.array)/trac  # Computing stress concentration factor
    logging.info(f"SCF for N = {N}:{SCF}")  # Logging SCF to terminal
    # if display is True diplay solution, stress components and von mises stress
    if display:
        ### Displaying solutions
        # Displaying displacement and micro-rotation using pv------------------------------------- 

        p = pv.Plotter()
        topology, cell_types, geometry = plot.vtk_mesh(V0)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        grid[""] = u.x.array.reshape(geometry.shape[0], 2)
        actor_0 = p.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_scalar("", factor=10)
        actor_1 = p.add_mesh(warped, show_edges=True)
        p.set_background("white")
        p.camera.tight(0.35, view='xy')
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show(auto_close=False, interactive=False)
            p.screenshot(img_dir + "/displacement_xy_view.png")
            p.close()

        p = pv.Plotter()
        topology, cell_types, geometry = plot.vtk_mesh(V1)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        grid[""] = θ.x.array.reshape(geometry.shape[0], 1)
        actor_0 = p.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_scalar("", factor=0.01)
        actor_1 = p.add_mesh(warped, show_edges=True)
        p.set_background("white")
        p.camera.tight(0.35, view='xy')
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show(auto_close=False, interactive=False)
            p.screenshot(img_dir + "/microrotation_xy_view.png")
            p.close()

        # von Mises stress
        warped.cell_data["VonMises"] = stresses.x.petsc_vec.array
        warped.set_active_scalars("VonMises")
        p = pv.Plotter()
        p.add_mesh(warped)
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show()

        ## Displaying sigma component and savings screen to a folder
        warped.cell_data["xx"] = σxx.x.petsc_vec.array
        warped.set_active_scalars("xx")
        p = pv.Plotter()
        p.add_mesh(warped, show_scalar_bar=True)
        p.set_background("white")
        p.camera.tight(0.35, view='xy')
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show(auto_close=False, interactive=False)
            p.screenshot(img_dir + "/sigma_xx.png")
            p.close()

        warped.cell_data["xy"] = σxy.x.petsc_vec.array
        warped.set_active_scalars("xy")
        p = pv.Plotter()
        p.add_mesh(warped)
        p.set_background("white")
        p.camera.tight(0.35, view='xy')
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show(auto_close=False, interactive=False)
            p.screenshot(img_dir + "/sigma_xy.png")
            p.close()

        warped.cell_data["yx"] = σyx.x.petsc_vec.array
        warped.set_active_scalars("yx")
        p = pv.Plotter()
        p.add_mesh(warped)
        p.set_background("white")
        p.camera.tight(0.35, view='xy')
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show(auto_close=False, interactive=False)
            p.screenshot(img_dir + "/sigma_yx.png")
            p.close()

        warped.cell_data["yy"] = σyy.x.petsc_vec.array
        warped.set_active_scalars("yy")
        p = pv.Plotter()
        p.add_mesh(warped)
        p.set_background("white")
        p.camera.tight(0.35, view='xy')
        p.show_axes()
        if not pv.OFF_SCREEN:
            p.show(auto_close=False, interactive=False)
            p.screenshot(img_dir + "/sigma_yy.png")
            p.close()
    
    ### Saving solution to a .pvd file if save is True
    if save:
        with io.VTKFile(MPI.COMM_WORLD, __file__.rstrip('.py') + "/displacement.pvd", "w") as vtk:
            vtk.write_function(u, 0.0)  # 0.0 = time step

        with io.VTKFile(MPI.COMM_WORLD, __file__.rstrip('.py') + "/micro_rotation.pvd", "w") as vtk:
            vtk.write_function (θ, 0.0)  # 0.0 = time step
    return SCF

if __name__ == "__main__":
    from sys import argv
    from pandas import DataFrame
    from pathlib import Path

    Path("Csv").mkdir(parents=True, exist_ok=True)
    Path("Immagini").mkdir(parents=True, exist_ok=True)
    SCF_list = []  # list to store SCF values

    if len(argv) != 1 and argv[1] == 'meshelement':
        for lenght in maxLengthElement:
            logging.info(f"Running multiple simulation with different mesh size")
            logging.info(f"Running simulation with maxElement = {lenght}")
            SCF = main(minElement=0.1*lenght, maxElement=lenght)
            SCF_list.append(SCF)
        df = DataFrame({'min_element': (minElement:= [n/10 for n in  maxLengthElement]),
                        'SCF': SCF_list})
        df.to_csv(path:="Csv/SCF_mesh", index=False)
        logging.info(f"Saving results to {path}")
        
    elif len(argv) != 1 and argv[1] == 'N':
        for N in N_values:
            logging.info(f"Running multiple simulation for different value of coupling number N")
            logging.info(f"Running simulation with N = {N}")
            SCF = main(N=N)
            SCF_list.append(SCF)
        df = DataFrame({'N': N_values,
                        'SCF': SCF_list})
        df.to_csv(path:="Csv/SCF_coupling_number2", index=False)
        logging.info(f"Saving results to {path}")
    else:
        SCF = main()
