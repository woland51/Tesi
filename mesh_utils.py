import gmsh
from pathlib import Path
from dolfinx.io import gmshio
from mpi4py import MPI

def plate_with_hole_field(D: float = 0.03,
                    L: float = 0.15,
                    maxLength: float = 0.005,
                    minLength: float = 0.0001,
                    save: bool = True,
                    file_name: str = "plate_with_hole",
                    directory: str = "mesh") -> None:
    """
    Create a 2D plate with a circular hole in the bottom left corner using GMSH.
    Parameters
    ----------
    D : float
        Diameter of the circular hole.

    L : float
        Length of the sides of the square plate.

    maxLength : float
        Maximum mesh size.

    minLength : float
        Minimum mesh size.

    file_name : str
        Name of the output mesh file.
    Returns : None, if save is set to False returns a dolfinx mesh object
    """

    Path(directory).mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.model.add(file_name)
    # Building a rectangle and a disk
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
    disk = gmsh.model.occ.addDisk(0, 0, 0, D/2, D/2)
    # Boolean difference between rectangle and disk
    plate, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])

    gmsh.model.occ.synchronize()

    # Tag surface
    gmsh.model.addPhysicalGroup(2, [plate[0][1]], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Plate_surf")

    # Tag boundary, useful for BCs in FEniCSx
    boundary = gmsh.model.getBoundary([plate[0]], oriented=False)
    boundary_curves = [b[1] for b in boundary if b[0] == 1]
    tags = [1,2,3,4]  # Etichette per i bordi
    gmsh.model.setPhysicalName(1, 2, "Boundary")

    for tag in tags:
        gmsh.model.addPhysicalGroup(1, [tag+1], tag=tag)
    # Mesh refinement: Distance field + Threshold
    # Find inner hole curves
    inner_curves = []
    for curve in boundary:
        c = curve[1]
        com = gmsh.model.occ.getCenterOfMass(1, c)
        # Se vicino all'origine, è il foro
        if (com[0]**2 + com[1]**2)**0.5 < D / 2:
            inner_curves.append(c)

    # Field 1: Distance from inner curves
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", inner_curves)

    # Field 2: Threshold for finer mesh near the hole
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", minLength)  # finer near the hole
    gmsh.model.mesh.field.setNumber(2, "SizeMax", maxLength)   # coarser away from the hole
    gmsh.model.mesh.field.setNumber(2, "DistMin", D / 10)
    gmsh.model.mesh.field.setNumber(2, "DistMax", D / 2)

    # Setting the field as a background
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # mesh_parameter
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", minLength)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxLength)

    gmsh.model.mesh.generate(2)
    if save:
        msh_path = "".join([directory,"/", file_name, ".msh"])
        gmsh.write(msh_path)
    else:
        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
        return domain, cell_markers, facet_markers
        
    gmsh.finalize()


def plate_with_hole_geo(D: float = 0.03,
                    L: float = 0.15,
                    maxLength: float = 0.005,
                    minLength: float = 0.0001,
                    save: bool = True,
                    file_name: str = "plate_with_hole",
                    directory: str = "mesh") -> None:
    """
    Create a 2D plate with a circular hole in the bottom left corner using GMSH.
    Parameters
    ----------
    D : float
        Diameter of the circular hole.

    L : float
        Length of the sides of the square plate.

    maxLength : float
        Maximum mesh size.

    minLength : float
        Minimum mesh size.

    file_name : str
        Name of the output mesh file.
    Returns : None, if save is set to False returns a dolfinx mesh object
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.model.add("plate_with_hole")
    gmsh.model.geo.addPoint(0, 0, 0, maxLength, tag=1)           # Hole center
    gmsh.model.geo.addPoint(0.015, 0, 0, minLength, tag=2)       # Point to the right of the hole
    gmsh.model.geo.addPoint(0, 0.015, 0, minLength, tag=3)       # Point to the left of the hole
    gmsh.model.geo.addPoint(0, 0.15, 0, maxLength, tag=4)        # Top-left
    gmsh.model.geo.addPoint(0.15, 0, 0, maxLength, tag=5)        # Bottom-right
    gmsh.model.geo.addPoint(0.15, 0.15, 0, maxLength, tag=6)     # Top right

    # Lines of the rectangle
    gmsh.model.geo.addLine(4, 3, tag=1)
    gmsh.model.geo.addLine(4, 6, tag=2)
    gmsh.model.geo.addLine(6, 5, tag=3)
    gmsh.model.geo.addLine(5, 2, tag=4)

    # Inner circle
    gmsh.model.geo.addCircleArc(3, 1, 2, tag=5)

    # Build the surface
    gmsh.model.geo.addCurveLoop([2, 3, 4, -5, -1], tag=1)
    gmsh.model.geo.addPlaneSurface([1], tag=1)

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [1], tag=1)
    gmsh.model.setPhysicalName(1, 1, "left")
    gmsh.model.addPhysicalGroup(1, [2], tag=2)
    gmsh.model.setPhysicalName(1, 2, "top")
    gmsh.model.addPhysicalGroup(1, [3], tag=3)
    gmsh.model.setPhysicalName(1, 3, "right")
    gmsh.model.addPhysicalGroup(1, [4], tag=4)
    gmsh.model.setPhysicalName(1, 4, "bottom")
    gmsh.model.addPhysicalGroup(2, [1], tag=5)
    gmsh.model.setPhysicalName(2, 5, "Plate_surf")

    # Local refinement
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", minLength)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxLength)

    gmsh.model.mesh.generate(2)

    if save:
        msh_path = "".join([directory,"/",  file_name, ".msh"])
        gmsh.write(msh_path)
    else:
        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
        return domain, cell_markers, facet_markers
        
    gmsh.finalize()
