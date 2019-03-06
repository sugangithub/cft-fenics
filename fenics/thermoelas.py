import fenics as fn
import numpy as np
import matplotlib.pyplot as plt
import mshr as ms

#   Defining parameters
l,h=5.0,0.3
T_top,T_bottom,T_left_right=fn.Constant(0.0),fn.Constant(50.0),fn.Constant(0.0)

#   Creating mesh
domain=ms.Rectangle(fn.Point(0.0,0.0),fn.Point(l,h))
mesh=ms.generate_mesh(domain,30)
#fn.plot(mesh,title='Mesh')
#plt.show()

#   Defining boundaries
def left_right(x,on_boundary):
    return (fn.near(x[0],0) or fn.near(x[0],l)) and on_boundary
def top(x,on_boundary):
    return (fn.near(x[1],h)) and on_boundary
def bottom(x,on_boundary):
    return (fn.near(x[1],0)) and on_boundary

#   Defining function spaces, test and trial function spaces
FS=fn.FunctionSpace(mesh,'CG',2)
T,v=fn.TrialFunction(FS),fn.TestFunction(FS)
T_sol=fn.Function(FS,name="Temperature rise")

#   Defining Dirichlet boundary conditions
bcs=[fn.DirichletBC(FS,T_top,top),fn.DirichletBC(FS,T_bottom,bottom),fn.DirichletBC(FS,T_left_right,left_right)]

#   Weak form formulation
lhs=fn.inner(fn.grad(T),fn.grad(v))*fn.dx
rhs=fn.Constant(0.0)*v*fn.dx

#   Solving weak formulation
fn.solve(lhs==rhs,T_sol,bcs)

#   Ploting
plt.figure()
p=fn.plot(T_sol,mode="contour")
plt.colorbar(p)
plt.show()

#   Defning mechanical parameters
E=fn.Constant(50.0e3)
nu=fn.Constant(0.2)
mu=E/2/(1+nu)
lm=E*nu/(1+nu)/(1-2*nu)
alpha=fn.Constant(1e-5)
rho_g=2400*9.81e-6

#   Defining sigma and symmetric part of a tensor
def eps(v):
    return fn.sym(fn.grad(v))
def sigma(v,T):
#def sigma2(v):
#    return (lm*fn.tr(eps(v)))*fn.Identity(2)+2.0*mu*eps(v)
    return (lm*fn.tr(eps(v))-alpha*(3*lm+2*mu)*T)*fn.Identity(2)+2.0*mu*eps(v)

#   Defining vector function spaces

VFS=fn.VectorFunctionSpace(mesh,'P',1)
f=fn.Constant((0.0,0.0))
u,w=fn.TrialFunction(VFS),fn.TestFunction(VFS)
u_sol=fn.Function(VFS,name="Displacement")

#   Boundary conditions
u_bcs=fn.DirichletBC(VFS,fn.Constant((0.0,0.0)),left_right)

#   Weak formulation
u_lhs=fn.inner(sigma(u,T_sol),eps(w))*fn.dx
#u_lhs=fn.inner(sigma2(u),eps(w))*fn.dx
u_rhs=fn.inner(f,w)*fn.dx
l_term=fn.lhs(u_lhs)
r_term=fn.rhs(u_lhs)+u_rhs

#   Solving
f.assign(fn.Constant((0.0,-rho_g)))
fn.solve(l_term==r_term,u_sol,u_bcs)

#   Plotting the Displacement
plt.figure()
p1=fn.plot(1e3*u_sol[1],title="Vertical Displacement[mm]")
plt.colorbar(p1)
plt.show()
plt.figure()
p2=fn.plot(sigma(u_sol,T_sol)[0,0],title="Horizontal stress[MPa]")
plt.colorbar(p2)
plt.show()
