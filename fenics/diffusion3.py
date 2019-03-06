import fenics as fn
import mshr as ms
import matplotlib.pyplot as plt
import numpy as np

#   Creating Mesh
a=1.0
b=2.0
alpha=0.5
timestep=0.05
diffusion=0.01
beta=10.0
alpha=40.0
gamma=0.2
domain=ms.Rectangle(fn.Point(0,0),fn.Point(a,b))
mesh=ms.generate_mesh(domain,10)
#fn.plot(mesh)
#plt.show()

SFS=fn.FunctionSpace(mesh,'P',1)
c0=fn.Function(SFS)
c=fn.Function(SFS)
v=fn.TestFunction(SFS)

#   Initial condition
f=fn.Expression('alpha*sin(2*PI*x[0])*cos(2*PI*x[1])',alpha=alpha,PI=np.pi,degree=1)
c0.interpolate(f) #mapping the expression on to our domain in function space
fn.plot(c0)
plt.show()

#   Boundary condition
def boundary(x,on_boundary):
    return on_boundary
g=fn.Expression('beta*x[0]*x[1]',beta=beta,degree=1)
#g=fn.Constant(2.0)
bc=fn.DirichletBC(SFS,g,boundary)

#   Weak form
h=fn.Expression('alpha*exp(-x[0]*x[0]-x[1]*x[1])',alpha=alpha,gamma=gamma,PI=np.pi,degree=1) #Source
#h=fn.Constant(0.0)
form=((fn.inner((c-c0)/timestep,v)+diffusion*(1-alpha)*fn.inner(fn.grad(c0),fn.grad(v))+diffusion*alpha*fn.inner(fn.grad(c),fn.grad(v)))-h*v)*fn.dx
time=0.0
#cFile=fn.File('conc.pvd')
#cFile<<(c0,time)

#   Loop for Solving
for ti in range(int(5.0/timestep)):
    fn.solve(form == 0,c,bc)
    c0.assign(c)
    time += timestep
#    cFile<<(c0,time)
fn.plot(c)
plt.show()
