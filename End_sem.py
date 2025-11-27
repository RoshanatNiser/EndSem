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

for t in range(0, 100001, 500):
    T.append(t)
    
    if t == 0:
        L.append(5000)
        R.append(0)
    else:
        D=pRNG(n=t)
        L_t=0
        
        start_idx = max(0, t - 5000)
        for i in range(start_idx, t):
            if D[i] < 0.5:
                L_t += 1
        
        R_t = 5000 - L_t
        L.append(L_t)
        R.append(R_t)

# Plots for question 1
plt.figure(figsize=(8,5))

plt.plot(T, L, marker='o', linestyle='-', label="No of particles in the left")
plt.plot(T, R, marker='o', linestyle='-', label="No of particles in the right")

plt.title("Question_1: No. of particles in left and right side of the wall", fontsize=18)
plt.xlabel("Time", fontsize=14)
plt.ylabel("No. of particles", fontsize=14)
plt.legend(fontsize=12)
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

N=1000
A= trap_int(t=42,a=0,b=2,N=N)
B= trap_int(t=41,a=0,b=2,N=N)

c=A/B
print(f"Center of mass is at x={c}meter\n")




#Question_5
print("Solving Question 5")

Y,V,T=RK4_2(k=5,a=0,b=12,t_i=0,t_f=1,v0=10)
print(f"Maximum Height reached by the object is {max(Y)}\n")

# Plots for question 5
plt.figure(figsize=(8,5))

plt.plot(Y, V, marker='o', linestyle='-')

plt.title("Question_5: Variation of the velocity with the height", fontsize=18)
plt.xlabel("Height (m)", fontsize=14)
plt.ylabel("Velocity (m/s)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('Question_5.png', dpi=300, bbox_inches="tight")
plt.close()

print("Figure save as Question_5.png\n")

#Question 6
print("Solving Question 6")

dx = 2.0/20
dt = 4.0/5000

U,X=PDE_H(L=2,dx=dx,dt=dt,T=5000)

# Plot temperature profiles at different time steps
plt.figure(figsize=(10, 6))

Time= [0,10,20,50,100,200,500,1000]
for i in Time:
    if i < len(U):
        plt.plot(X, U[i], marker='o', label=f't = {i}', markersize=4)

plt.title('Question 6: Heat Equation - Temperature Evolution')
plt.xlabel('Position x (m)')
plt.ylabel('Temperature T(x) (°C)')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig("Question_6.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'Question_6.png\n")

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
