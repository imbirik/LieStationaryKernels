import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay


def save_points(points, colors, space_name, color_source, triangulation=None, boundary_v=None):
    if triangulation is not None:
        triangulation = triangulation
    elif points.shape[1] == 3:
        triangulation = ConvexHull(points)
    elif points.shape[1] == 2:
        triangulation = Delaunay(points)
    with open(space_name + "_points.obj", "w") as f:
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
                f.write(template_f(tri[0] + 1, tri[1] + 1, tri[2] + 1))

    pd.DataFrame(colors).to_csv(space_name+"_"+ color_source + "_colors.csv", index=False, header=False)