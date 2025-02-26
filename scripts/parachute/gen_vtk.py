import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import vtk

file = "Ribbon_median2_"
##load displacement data 
data = np.load("data/"+file+"data.npz")

node_mask, nodes = data["node_mask"], data["nodes"]
node_weights = data["node_weights"]
directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
y, x, out= data["y"], data["x"], data["out"]

points = nodes.reshape((14474,3))
node_mask = node_mask.reshape(14474,)

##We only need the predicted solution “out ”and the actual solution "y".
out, y = out.reshape(14474,12), y.reshape(14474,12)

##vtk needs element data
elems = np.load("data/"+file+"elem.npy")
elems_num = elems.shape[0]
num = np.count_nonzero(node_mask)
print(np.linalg.norm(out.reshape(-1)-y.reshape(-1),2))
print(np.linalg.norm(out-y,np.inf))
points = points[:num,:]

vtk_points = vtk.vtkPoints()
points0 = points
for i in range(num):
    point = points0[i,:]
    vtk_points.InsertNextPoint(point)

##Create a vtkPolyData object and set the point data.
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)

##Create a vtkPolyData object and set the cell data.
lines = vtk.vtkCellArray()
triangles = vtk.vtkCellArray()

for j in range(elems_num):
    cell_dim = elems[j,0]
    cell_index = elems[j,1:]

    if cell_dim == 1:
        #line
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, cell_index[1])
        line.GetPointIds().SetId(1, cell_index[2])
        lines.InsertNextCell(line)

        
    if cell_dim == 2:
        #triangle
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetNumberOfIds(3)
        triangle.GetPointIds().SetId(0, cell_index[0])
        triangle.GetPointIds().SetId(1, cell_index[1])
        triangle.GetPointIds().SetId(2, cell_index[2])
        triangles.InsertNextCell(triangle)

polydata.SetPolys(triangles)
polydata.SetLines(lines)

##Generate the initial state.
writer = vtk.vtkPolyDataWriter()
writer.SetFileName("data_vtk/Ribbon/displacements_"+file+str(10*(0))+".vtk")
writer.SetInputData(polydata)
writer.SetFileTypeToBinary()
writer.Write()


for l in range(4): 
##Generating subsequent data.
    predict_displacements = out[:num,3*l:3*l+3]
    true_displacements = y[:num, 3*l:3*l+3]

    ##导入格点数据
    vtk_points = vtk.vtkPoints()
    points0 = points + true_displacements
    for i in range(num):
        point = points0[i,:]
        vtk_points.InsertNextPoint(point)

    ##Create a vtkPolyData object and set the point data.
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    ##Create a vtkPolyData object and set the cell data.
    lines = vtk.vtkCellArray()
    triangles = vtk.vtkCellArray()

    for j in range(elems_num):
        cell_dim = elems[j,0]
        cell_index = elems[j,1:]

        if cell_dim == 1:
            #线
            line = vtk.vtkLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, cell_index[1])
            line.GetPointIds().SetId(1, cell_index[2])
            lines.InsertNextCell(line)

        
        if cell_dim == 2:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetNumberOfIds(3)
            triangle.GetPointIds().SetId(0, cell_index[0])
            triangle.GetPointIds().SetId(1, cell_index[1])
            triangle.GetPointIds().SetId(2, cell_index[2])
            triangles.InsertNextCell(triangle)

    polydata.SetPolys(triangles)
    polydata.SetLines(lines)

    # Create a vtkFloatArray object to store 3D displacement data.

    #error
    displacements = vtk.vtkFloatArray()
    displacements.SetName("Displacements error")  # 设置属性名称
    displacements.SetNumberOfComponents(3)  # 设置分量数为3
    displacements.SetNumberOfTuples(num) # 设置数组长度
    for i in range(num):
        displacements.SetTuple(i, abs(true_displacements[i,:]-predict_displacements[i,:]))
    polydata.GetPointData().AddArray(displacements)

     #生成数据
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("data_vtk/Ribbon/displacements_"+file+str(10*(l+1))+".vtk")
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()

