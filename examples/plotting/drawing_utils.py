import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay

def save_points(points, colors, boundary_v=None, filename=None):
    if points.shape[1] == 3:
        triangulation = ConvexHull(points)
    elif points.shape[1] == 2:
        triangulation = Delaunay(points)
    with open(filename+"_points.obj", "w") as f:
        f.write("o Shape_IndexedFaceSet\n")
        template_v = lambda x, y, z=0: "v {} {}  {} \n".format(x, y, z)
        for point in points:
            f.write(template_v(*tuple(x for x in point)))
        f.write("s off \n")
        template_f = lambda x, y, z: "f {} {}  {} \n".format(x, y, z)
        for tri in triangulation.simplices:
            if boundary_v is not None:
                if (tri[0] not in boundary_v) or (tri[1] not in boundary_v) or (tri[2] not in boundary_v):
                    f.write(template_f(tri[0] + 1, tri[1] + 1, tri[2] + 1))
                else:
                    print("skipped")
            else:
                f.write(template_f(tri[0] + 1, tri[1] + 1, tri[2] + 1))

    pd.DataFrame(colors).to_csv(filename+"_colors.csv", index=False, header=False)


def save_points_with_color(points, colors, boundary_v=None, filename=""):
    c_min, c_max = colors.min(), colors.max()
    colors = (colors-c_min)/(c_max-c_min)
    if points.shape[1] == 3:
        triangulation = ConvexHull(points)
    elif points.shape[1] == 2:
        triangulation = Delaunay(points)
    with open(filename + "_points_colors.obj", "w") as f:
        f.write("o Shape_IndexedFaceSet\n")
        template_v = lambda x, y, z=0, c=0: "v {} {}  {} {} 0 0\n".format(x, y, z, c)
        for point, color in zip(points, colors):
            f.write(template_v(*tuple(x for x in point), c=color))
        f.write("s off \n")
        template_f = lambda x, y, z: "f {} {}  {} \n".format(x, y, z)
        for tri in triangulation.simplices:
            if boundary_v is not None:
                if (tri[0] not in boundary_v) or (tri[1] not in boundary_v) or (tri[2] not in boundary_v):
                    f.write(template_f(tri[0] + 1, tri[1] + 1, tri[2] + 1))
                else:
                    print("skipped")
            else:
                f.write(template_f(tri[0] + 1, tri[1] + 1, tri[2] + 1))

