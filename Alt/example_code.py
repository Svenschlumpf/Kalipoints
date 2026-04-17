
#___________________________________________________________________________________________________________________
# Nr. 1
# Source - https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
# Posted by Fnord, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-05, License - CC BY-SA 4.0
# Further links:
#   documentation ->    https://arxiv.org/pdf/0912.4540
#   visualisation ->    https://openprocessing.org/sketch/41142

import math

def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


print(fibonacci_sphere())



#__________________________________________________________________________________________________________________
# Nr. 2
# Source - https://stackoverflow.com/questions/70977042/how-to-plot-spheres-in-3d-with-plotly-or-another-library
# Posted by Stéphane Laurent
# Retrieved 2025-12-05, License - CC BY-SA 4.0

import numpy as np
import pyvista as pv
import plotly.graph_objects as go
import plotly.io as pio # to save the graphics as html

# center 
x0 = 0
y0 = 0
z0 = 0
# radius 
r = 2
# radius + 2% to be sure of the bounds
R = r * 1.02

def f(x, y, z):
    return (
        (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    )

# generate data grid for computing the values
X, Y, Z = np.mgrid[(-R+x0):(R+x0):250j, (-R+y0):(R+y0):250j, (-R+z0):(R+z0):250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order = "F")
# compute the isosurface f(x, y, z) = r²
isosurf = grid.contour(isosurfaces = [r**2])
mesh = isosurf.extract_geometry()
# extract vertices and triangles
vertices = mesh.points
triangles = mesh.faces.reshape(-1, 4)

# plot
fig = go.Figure(data=[
    go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        colorscale = [[0, 'gold'],
                     [0.5, 'mediumturquoise'],
                     [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity = np.linspace(0, 1, len(vertices)),
        # i, j and k give the vertices of the triangles
        i = triangles[:, 1],
        j = triangles[:, 2],
        k = triangles[:, 3],
        showscale = False
    )
])

fig.show()

# save
pio.write_html(fig, "plotlySphere.html")
