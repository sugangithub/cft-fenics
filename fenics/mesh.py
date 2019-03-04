from dolfin import *
import dolfin as df
import mshr as ms
import matplotlib.pyplot as plt
import numpy as np

#   Testing all the meshes

#   1.Unit interval mesh
mesh1=UnitIntervalMesh(10)
df.plot(mesh1,title="Unit Interval")
#plt.show()

#   2.Square mesh
mesh2=UnitSquareMesh(10,10,"crossed")
df.plot(mesh2,title="Square crossed mesh")
#plt.show()

#   3.Ellipse mesh
#mesh3=EllipseMesh(Point(0.0,0.0),[3.0,1.0],0.2)
domain=ms.Ellipse(df.Point(0.0,0.0),2,1)
mesh3=ms.generate_mesh(domain,30)
df.plot(mesh3,title="Ellipse mesh")
#plt.show()

#   4.Unit cube mesh1
mesh4=UnitCubeMesh(10,10,10)
df.plot(mesh4,"unit cube")
#plt.show()

#   5.Box mesh
mesh5=BoxMesh(0.0,0.0,0.0,10.0,2.0,4.0,5,5,5)
df.plot(mesh5,title="Box mesh")
plt.show()
