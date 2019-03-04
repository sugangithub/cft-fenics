from fenics import *
import mshr as ms
import matplotlib.pyplot as plt
import numpy as np

T=2.0               #final times
num_steps=10        #no of time steps
dt=T/num_steps      #time steps size
alpha=2.0
beta=1.2

#   Creating mesh and function space
nx=8
ny=8
mesh=UnitSquareMesh(nx,ny)
V=FunctionSpace(mesh,'P',1)

#   Defining boundary conditions
u_D=Expression('1',degree=2,alpha=alpha,beta=beta,t=0)
def boundary(x,on_boundary):
    return on_boundary

bc=DirichletBC(V,u_D,boundary)

#   Define intial value
u_n=interpolate(u_D,V)

#   Define Variational Problem
u=TrialFunction(V)
v=TestFunction(V)
f=Constant(beta-2-2*alpha)

F=u*v*dx+dt*dot(grad(u),grad(v))*dx-(u_n+dt*f)*v*dx
a=lhs(F)
L=rhs(F)

#   Time stepping
u=Function(V)
t=0
for n in range(num_steps):
    #   Update time
    t += dt
    u_D.t = t # TEMP

    #   Complete solution
    solve(a==L,u,bc)

    u_n.assign(u)

plot(u)
plt.show()
