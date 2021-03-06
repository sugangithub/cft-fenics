import fenics as fn
import mshr as ms
import numpy as np
import matplotlib.pyplot as plt

a=1.0
b=2.0

T1=10.0
T2=10.0
bar=ms.Rectangle(fn.Point(0,0),fn.Point(a,b))
hole=ms.Circle(fn.Point(a/2,b/2),a/6)
domain=bar-hole
mesh=ms.generate_mesh(domain,20)

tol=1.0e-14
def bottom(x,on_boundary):
    return on_boundary and x[1]<tol

def top(x,on_boundary):
    return on_boundary and b-x[1]<tol

FS=fn.FunctionSpace(mesh,'P',1)
v=fn.TestFunction(FS)
u=fn.TrialFunction(FS)

bc_top=fn.DirichletBC(FS,T1,top)
bc_bottom=fn.DirichletBC(FS,T2,bottom)

bcs=[bc_top,bc_bottom]

form=fn.dot(fn.grad(u),fn.grad(v))*fn.dx
u_sol=fn.Function(FS)
f = fn.Expression('100*sin(x[0])',degree=1)
g=fn.Expression('100*cos(x[0])',degree=1)
L = f*v*fn.dx+g*v*fn.ds

A,c=fn.assemble_system(form,L,bcs) # Creating a system Ax=c
d1=fn.PointSource(FS,fn.Point(a/2.0,b/5.0),100)
d2=fn.PointSource(FS,fn.Point(a/2.0,4*b/5.0),100)
d1.apply(c)
d2.apply(c)

fn.solve(A,u_sol.vector(),c)


fig,ax=plt.subplots()
cs=fn.plot(u_sol)
fig.colorbar(cs)
plt.show()
