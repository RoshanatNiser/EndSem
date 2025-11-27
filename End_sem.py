# Name: Roshan Yadav
# Roll no: 2311144
# End_Sem_exam_2025


# Imports
from Function_lib import *
import math

# Question_1
print("Question no 1\n")
L=[]
R=[]
T=[]
for t in range(5000,10000,500):
    T.append(t)
    D=pRNG(n=t)
    L_t=[]
    R_t=[]
    for i in range(0,5000):
        if D[-i] < 0.5:
            L_t.append(1)
        else:
            R_t.append(1)
    
    L.append(sum(L_t))
    R.append(sum(R_t))

# Plots for question 1
plt.figure(figsize=(8,5))

plt.plot(T, L, marker='o', linestyle='-', label="No of particles in the left")
plt.plot(T, R, marker='o', linestyle='-', label="No of particles in the right")

plt.title("Question_1: No. of particles in left and right side of the wall", fontsize=18)
plt.xlabel("Time", fontsize=14)
plt.ylabel("No. of particles", fontsize=14)
plt.legend(False)
plt.savefig('Question_1.png', dpi=300, bbox_inches="tight")
plt.close()

print("Figure save as Question_1.png\n")


# Question_2
print("Solving Question 2")
M=[[4,-1,0,-1,0,0],[-1,4,-1,0,-1,0],[0,-1,4,0,0,-1],[-1,0,0,4,-1,0],[0,-1,0,-1,4,-1],[0,0,-1,0,-1,4]]
b=[2,1,2,2,1,2]
A=matrix(M)
X=A.gauss_seidel(b,tol=10**-6)
print(f"The solution to the linear equation is {X}\n")



#Question_3
print("Solving Question no 3")
x=newton_raphson(y0=0,x0=0,t=3,k=32)
print(f"The answer t0 question 3 is {x}\n")



#Question 4
print("Solving Question 4")
""" We known centre of mass (c)= (integral xdm from x=0 to x=2)/ (integral dm from x=0 to x=2).
so  c= (integral x^3dx from x=0 to x=2)/ (integral x^2 dx from x=0 to x=2) because dm= lamda(x)dx= x^2dx.

Let A=(integral x^4dx from x=0 to x=2)
    B=(integral x^2 dx from x=0 to x=2).
    
    This intergration is done using trapedoial method using N points"""

N=1000
A= trap_int(t=42,a=0,b=2,N=N)
B= trap_int(t=41,a=0,b=2,N=N)

c=A/B
print(f"Center of mass is at x={c}meter\n")




#Question_5
print("Solving Question 5")

Y,V,T=RK4_2(k=5,a=0,b=12,t_i=0,t_f=1)
print(f"Maximum Height reached by the object is {max(Y)}\n")
# Plots for question 5
plt.figure(figsize=(8,5))

plt.plot(Y, V, marker='o', linestyle='-', label=None)


plt.title("Question_5: Variation of the velocity with the height", fontsize=18)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Velocity", fontsize=14)
plt.legend(fontsize=12)
plt.savefig('Question_5.png', dpi=300, bbox_inches="tight")
plt.close()

#Question 6
print("Solving Question 6")

"""Given nx=20 and L=2 implies dx=1/20=0.005. And given nt=5000 and 0<=t<=4 implies dt=1/5000=0.0002"""

U,X=PDE_H(L=2,dx=0.05,dt=0.0005,T=1000)

# Plot temperature profiles at different time steps
plt.figure(figsize=(10, 6))

Time= [0,10,20,50,100,200,500,1000]
for i in Time:
    if i < len(U):
        plt.plot(X, U[i], marker='o', label=f't = {i}', markersize=4)

plt.title('Question 2: Heat Equation - Temperature Evolution')
plt.xlabel('Position x (m)')
plt.ylabel('Temperature T(x) (°C)')
plt.legend(False)
plt.grid(True, alpha=0.3)
plt.savefig("Question_6.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'Assgn_15_Question_2.png\n")

#Question 7
print("Solving question 7")

data=read_matrix("esem4fit.txt")
X=[]
V=[]
for i in range(len(data)):
    X.append(data[i][0])
    V.append(data[i][1])

A=polynomial_fitting(X=X, Y=V, k=4)

print(f"The Fitted set of Coefficient are {A}\n")


"""
Outputs:
Question no 1

Figure save as Question_1.png

Solving Question 2
Converged in 16 iterations
The solution to the linear equation is [0.9999997530614102, 0.9999997892247294, 0.9999999100460266, 0.9999998509593768, 0.9999998727858708, 0.9999999457079743]

Solving Question no 3
The answer t0 question 3 is 2.5538872639694894

Solving Question 4
Center of mass is at x=1.5000007499996224meter

Solving Question 5
Maximum Height reached by the object is 0

Solving Question 6
Plot saved as 'Assgn_15_Question_2.png

Solving question 7
The Fitted set of Coefficient are [0.25462950721154565, -1.193759213809225, -0.4572554123829642, -0.8025653910658196, 0.013239427477395338]
"""