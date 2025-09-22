import os
import numpy as np
import vtk
from vtk.util import numpy_support as VN

def read_vti(path):
    """
    Read a VTI (VTK ImageData) file and return:
    - data object
    - dimensions (nx, ny, nz)
    - array name
    - scalar values as numpy array
    - x, y, z coordinates arrays
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    dims = data.GetDimensions()  # (nx, ny, nz)
    arr_name = data.GetPointData().GetArrayName(0)
    vals = data.GetPointData().GetArray(arr_name)
    vals_np = VN.vtk_to_numpy(vals)

    # gather coordinates
    x = np.zeros(data.GetNumberOfPoints())
    y = np.zeros(data.GetNumberOfPoints())
    z = np.zeros(data.GetNumberOfPoints())
    for i in range(data.GetNumberOfPoints()):
        x[i], y[i], z[i] = data.GetPoint(i)

    return data, dims, arr_name, vals_np, x, y, z

def write_vtp_from_samples(stencil, vals_np, x, y, z, name, extra_indices, filename):
    """
    Save sampled points as a VTP (VTK PolyData) file.
    """
    pts = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    selected = np.where(stencil > 0)[0]
    for i in selected:
        pts.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # Add boundary anchors
    for j in extra_indices:
        pts.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().AddArray(val_arr)

    writer = vtk.vtkXMLPolyDataWriter()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    writer.SetFileName(filename)
    writer.SetInputData(poly)
    writer.Write()

def write_reconstruction_vti(data, recon_np, out_path, array_name='recon'):
    """
    Write reconstructed scalar values into the original VTI file as a new array.
    """
    vtk_arr = VN.numpy_to_vtk(recon_np)
    vtk_arr.SetName(array_name)
    data.GetPointData().AddArray(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(out_path)
    writer.Write()
