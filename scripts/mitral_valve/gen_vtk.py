import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import vtk

def add_array(name, data, num_components=3):
    
    array = vtk.vtkFloatArray()
    array.SetName(name)
    array.SetNumberOfComponents(num_components)
    for row in data:
        array.InsertNextTuple(row)
    return array 
        


def write_polydata_vtk(points, elems):
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    # Create vtkPolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    elems_num = elems.shape[0]

    lines = vtk.vtkCellArray()
    polygons = vtk.vtkCellArray()
    for j in range(elems_num):
        cell_dim = elems[j,0]
        cell_index = elems[j,1:]
        cell_nodes_sum = len(cell_index)
        if cell_dim == 1:
            #line
            line = vtk.vtkLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, cell_index[1])
            line.GetPointIds().SetId(1, cell_index[2])
            lines.InsertNextCell(line)

        if cell_dim == 2:
            #triangle
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(cell_nodes_sum)
            for i in range(cell_nodes_sum):
                polygon.GetPointIds().SetId(i, cell_index[i])
            polygons.InsertNextCell(polygon)

        polydata.SetPolys(polygons)

    return polydata




data_path = "../../data/mitral_valve/single_geometry"
##load displacement data 
i=1 

displacement = np.load(data_path+"/displacements_%05d"%(i+1)+".npy")
strain = np.load(data_path+"/lagrange_strain_%05d"%(i+1)+".npy")[:,[0,1,2,4,5,8]]  # xx xy xz ; yx yy yz ; zx zy zz
stress = np.load(data_path+"/cauchy_stress_%05d"%(i+1)+".npy")[:,[0,1,2,4,5,8]]



deformed_points = np.load(data_path+"/coordinates_%05d"%(i+1)+".npy") 
points = deformed_points - displacement
points_num = points.shape[0]

##vtk needs element data
elem_dim = 2
elems = np.load(data_path+"/quad_connectivity_%05d"%(i+1)+".npy")
elems = np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1)     
elems_num = elems.shape[0]


polydata = write_polydata_vtk(points, elems)
displacement_array = add_array("Displacement", displacement, num_components=3)
strain_array = add_array("Strain", strain, num_components=6)
stress_array = add_array("Stress", stress, num_components=6)

polydata.GetPointData().AddArray(displacement_array)
polydata.GetPointData().AddArray(strain_array)
polydata.GetPointData().AddArray(stress_array)



##Generate the initial state.
writer = vtk.vtkPolyDataWriter()
writer.SetFileName("mitral_valve_%05d"%(i+1)+".vtk")
writer.SetInputData(polydata)
writer.SetFileTypeToBinary()
writer.Write()


