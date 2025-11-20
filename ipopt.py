import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, Expression, value, minimize, RangeSet
import mat_4bus

COST = 1e-6
Ybus = mat_4bus.create_Ybus()
M_np = mat_4bus.get_GBBG(Ybus)

def objective(t, v):
    return (v[0]/t[0] - 1)**2 + (v[1]/t[1] - 1)**2 + (v[2]/t[2] - 1)**2 + (v[3]/t[3] - 1)**2

def solve(target):
    P2_target,  P3_target,  Q2_target,  Q3_target = target   

    model = ConcreteModel()

    V_BOUND = 5.0 
    # Vars: v = [V1_re, V2_re, V3_re, V4_re, V2_im, V3_im, V4_im]
    N = 7
    model.I = RangeSet(0, N-1)
    model.v = Var(model.I, bounds=(-V_BOUND, V_BOUND), initialize=1.0)  # init near 1.0 helps solver
    model.v[4] = 0.05
    model.v[5] = 0.05
    model.v[6] = 0.0

    # Create Expressions for currents i_j = sum_k M[j,k] * v_k
    def make_current_expr(m, j):
        return sum(float(M_np[j, k]) * m.v[k] for k in range(N))

    model.i = Expression(RangeSet(0, N-1), rule=make_current_expr)  # indices offset hack
    # Define P and Q expressions for bus 2 and 3 using P = Vr*Ir + Vi*Ii, Q = Vi*Ir - Vr*Ii; using I^*
    model.P2 = Expression(expr=model.v[1] * model.i[1] - model.v[4] * model.i[4])
    model.Q2 = Expression(expr=model.v[4] * model.i[1] + model.v[1] * model.i[4])
    model.P3 = Expression(expr=model.v[2] * model.i[2] - model.v[5] * model.i[5])
    model.Q3 = Expression(expr=model.v[5] * model.i[2] + model.v[2] * model.i[5])
    model.P1 = Expression(expr=model.v[0] * model.i[0])
    model.P4 = Expression(expr=model.v[3] * model.i[3] - model.v[6] * model.i[6])

    def objective_rule(m):
        return objective(target, [m.P2, m.P3, m.Q2, m.Q3]) - COST * (model.P1 + model.P4)
        #return (m.P2 / P2_target - 1)**2 + (m.Q2 / Q2_target - 1)**2 + (m.P3 / P3_target - 1)**2 + (m.Q3 / Q3_target - 1)**2

    model.obj = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory('ipopt')
    # solver.options['tol'] = 1e-8
    results = solver.solve(model, tee=True)  # tee=True prints solver log

    print("\nSolver termination condition:", results.solver.termination_condition)
    print("Objective value (sum squared mismatch):", value(model.obj))

    v_sol = np.array([value(model.v[i]) for i in range(N)])
    i_sol = M_np.dot(v_sol)
    print("\nVoltage solution vector:", v_sol)
    print("\nCurrent conj vector:", i_sol)

    # Compute P2,Q2,P3,Q3
    P2 = value(model.P2)
    Q2 = value(model.Q2)
    P3 = value(model.P3)
    Q3 = value(model.Q3)

    print(f"P2 computed = {P2:.6f}, target = {P2_target}, err2 = {np.abs(P2-P2_target):.6f}")
    print(f"Q2 computed = {Q2:.6f}, target = {Q2_target}, err2 = {np.abs(Q2-Q2_target):.6f}")
    print(f"P3 computed = {P3:.6f}, target = {P3_target}, err2 = {np.abs(P3-P3_target):.6f}")
    print(f"Q3 computed = {Q3:.6f}, target = {Q3_target}, err2 = {np.abs(Q3-Q3_target):.6f}")
    print(solve_powers(v_sol))

def solve_powers(V):
    Ic = M_np @ V
    S = np.zeros(V.shape)
    Vr, Vi = V[1:4], V[4:]
    Ir, Ii = Ic[1:4], Ic[4:]
    S[0] = Ic[0] * V[0] # slack bus Vi=0
    S[1:4] = Ir * Vr - Ii * Vi
    S[4:7] = Ir * Vi + Ii * Vr
    print("1:", S[0])
    for i in range(1, 4):
        print(f"{i+1}:", S[i], S[i+3])

if __name__ == "__main__":
    target = np.array((1.800, 1.700, 1.100, 1.050)) * 1
    solve(target)
