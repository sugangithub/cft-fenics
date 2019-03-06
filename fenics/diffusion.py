import fenics as fn
import mshr as ms
import matplotlib.pyplot as plt
import numpy as np

#   Creating Mesh
a=1
b=2
alpha=0.5
timestep=0.05
diffusion=0.1
domain=ms.Ellipse(fn.Point(0,0),b,a)
mesh=ms.generate_mesh(domain,20)
#fn.plot(mesh)
#plt.show()

SFS=fn.FunctionSpace(mesh,'P',1)
c0=fn.Function(SFS)
c=fn.Function(SFS)
v=fn.TestFunction(SFS)

#   Initial condition
ci=fn.Expression('sin(2*PI*x[0])*cos(2*PI*x[1])',PI=np.pi,degree=1)
c0.interpolate(ci)
#fn.plot(c0)
#plt.show()

#   Weak form
form=(fn.inner((c-c0)/timestep,v)+diffusion*(1-alpha)*fn.inner(fn.grad(c0),fn.grad(v))+diffusion*alpha*fn.inner(fn.grad(c),fn.grad(v)))*fn.dx

time=0.0
#cFile=fn.File('conc.pvd')
#cFile<<(c0,time)

#   Loop for Solving

for ti in range(int(1.0/timestep)):
    fn.solve(form == 0,c)
    c0.assign(c)
    time += timestep
    #cFile<<(c0,time)

fn.plot(c)
plt.show(c)
