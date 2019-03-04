#   Deflection of a membrane from a Load
from mshr import *
from dolfin import *
import matplotlib.pyplot as plt
from numpy import *

#   Creating the Mesh
domain = Circle(Point(0,0),1.2)
mesh=generate_mesh(domain,64)

#   Defining the parameters
beta=2
R0=1.0
R1=0.4
p=Expression('4*exp(-pow(beta,2)*(pow(x[0]-R0,2)+pow(x[1]-R1,2)))',degree=1,beta=beta,R0=R0,R1=R1)

#   Boundary
def boundary(x,on_boundary):
    return on_boundary

#   Defining the variational problem
V=FunctionSpace(mesh,'P',1)
w=TrialFunction(V)
v=TestFunction(V)
a=dot(grad(w),grad(v))*dx
L=p*v*dx


#   Defning Boundary conditions
bc=DirichletBC(V,Constant(0.0),boundary)

#   Defining the equation to solve
w=Function(V)
solve(a==L,w,bc)

#   Plotting the solution
p=interpolate(p,V)
plot(w,title="Deflection")
plt.show()
plot(p,title="Load")
plt.show()

#   Curve Plot
import numpy as np
tol=0.001
y=np.linspace(-1+tol,1-tol,101)
points=[(0,y_) for y_ in y]
w_line=np.array([w(point) for point in points])
p_line=np.array([p(point) for point in points])
plt.plot(y,50*w_line,'k',linewidth=2)
plt.plot(y,p_line,'b--',linewidth=2)
plt.grid(True)
plt.xlabel('$y$')
plt.legend(['Deflection','Load'],loc='upper left')
plt.show()
