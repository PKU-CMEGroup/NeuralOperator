import meshio
import numpy as np

i = 0
nodes = np.load("../../data/car_shapenet/nodes_%05d.npy"%(i))
elems = np.load("../../data/car_shapenet/elems_%05d.npy"%(i))
features = np.load("../../data/car_shapenet/features_%05d.npy"%(i))



# Nodal values: scalar (e.g., temperature) and vector (e.g., displacement)
nodal_scalar = features[:, 0]  # Scalar values at each node
nodal_vector = features[:, 1:] # Vector values

# Convert nodes to meshio-compatible format
points = nodes  # Extract (x, y, z) coordinates

# Convert elements to meshio-compatible format
cells = []
cells.append(("triangle", elems[:,1:]))

# Create the mesh
mesh = meshio.Mesh(
    points=points,
    cells=cells,
    point_data={
        "pressure": nodal_scalar,
        "normal": nodal_vector,
    }
)

# Save to an Exodus II file
meshio.write("car.vtk", mesh)