import fenics as fn
import mshr as ms
import matplotlib.pyplot as plt
import numpy as np

#print fn._version_

lm=10.0
mu=20.0

def eps(u):
    return fn.sym(fn.grad(u))

def sigma(u):
    return lm*(fn.tr(eps(u))*fn.Identity(3)) + 2*mu*eps(u)

w=1.0
h=1.0
l=5.0
domain=ms.Box(fn.Point(0,0,-w/2.0),fn.Point(l,h,w/2.0))
mesh=ms.generate_mesh(domain,20)
#fn.plot(mesh)
#plt.show()

#meshFile =fn.File('mesh_box.pvd')
#meshFile<<mesh

VFS=fn.VectorFunctionSpace(mesh,'P',1)

tol=1.0e-12
def left(x,on_boundary):
    return on_boundary and x[0]<tol

bc=fn.DirichletBC(VFS,fn.Constant((0,0,0)),left)

#   Gravity
g=0.1
f_ext=fn.Constant((0,-g,0))

u=fn.TrialFunction(VFS)
v=fn.TestFunction(VFS)

lhs=fn.inner(sigma(u),eps(v))*fn.dx
rhs=fn.inner(f_ext,v)*fn.dx

u_sol=fn.Function(VFS)
fn.solve(lhs==rhs,u_sol,bc)

#fn.plot(u_sol)
#plt.show()

uFile=fn.File('elastica.pvd')
uFile<<u_sol



#   Weak Formulation
