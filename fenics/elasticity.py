import fenics as fn
import mshr as ms
import matplotlib.pyplot as plt

#   Variables lambda and mu for constitutive relations
lm=1.0
mu=2.0

#   Giving symmetric part of gradient of input vector
def eps(u):
    return fn.sym(fn.grad(u))

#   Returns stress tensor
def sigma(u):
    return lm*(fn.tr(eps(u))*fn.Identity(2)) + 2*mu*eps(u)

#   Define the Mesh for rectangle
L=10.0
H=2.0
domain=ms.Rectangle(fn.Point(0,0),fn.Point(L,H))
mesh=ms.generate_mesh(domain,10)

fn.plot(mesh)
plt.show()

#   Define a vector function space in our mesh
VFS=fn.VectorFunctionSpace(mesh, 'P',2)

#   Defining Boudnary condns
tol=1.0e-14
def left(x,on_boundary):
    return on_boundary and x[0]<tol

#   Assigning diricichlet B.C to the problem
bc=fn.DirichletBC(VFS,fn.Constant((0,0)),left);

#   Giving external force as Gravity
fext=fn.Expression(('0','-0.03'),degree=1)

#   Define trial and test functions
u=fn.TrialFunction(VFS)
v=fn.TestFunction(VFS)

#   Equation to solve
lhs=fn.inner(eps(v),sigma(u))*fn.dx
rhs=fn.inner(v,fext)*fn.dx
u=fn.Function(VFS)

#   Solving the equation
fn.solve(lhs==rhs,u,bc)

#   Plotting the Solution
#fn.plot(u)
#plt.show()

print(u(9,1))
