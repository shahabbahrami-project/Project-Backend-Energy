import numpy as np
import cvxpy as cp
def MPC(Tindoor, Tout, Tdes, N, price, w):
    Tair=cp.Variable(pos=True)
    Tset=cp.Variable(pos=True)
    Tin=cp.Variable(pos=True)
    constraints = [
    Tair <= 30,
    Tair >= 10,
    Tset<=30,
    Tset >=0,
    Tin <= 30,
    Tin >= 0,
    Tin==Tindoor+np.exp(-300/130)*(Tout-Tindoor)+np.exp(-300/130)*(Tair-Tindoor),
    ]
    prob = cp.Problem(cp.Minimize(price*cp.norm(Tair-Tindoor, 1)+w*N*cp.norm(Tin-Tdes, 1)),
                      constraints)

    prob.solve(verbose=True, warm_start=True)

    return Tair.value, Tin.value, Tset.value, prob.value
