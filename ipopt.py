import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, Expression, value, minimize, RangeSet
import mat_4bus

Ybus = mat_4bus.create_Ybus()
M_np = mat_4bus.get_GBBG(Ybus)

def solve(target):
    P2_target,  P3_target,  Q2_target,  Q3_target = target   

    model = ConcreteModel()

    V_BOUND = 5.0   # per-unit style; adjust as needed
    # Vars: v = [V1_re, V2_re, V3_re, V4_re, V2_im, V3_im, V4_im]
    N = 7
    model.I = RangeSet(0, N-1)
    model.v = Var(model.I, bounds=(-V_BOUND, V_BOUND), initialize=0.9)  # init near 1.0 helps solver

    # Create Expressions for currents i_j = sum_k M[j,k] * v_k
    def make_current_expr(m, j):
        return sum(float(M_np[j, k]) * m.v[k] for k in range(N))

    model.i = Expression(RangeSet(0, N-1), rule=make_current_expr)  # indices offset hack
    # Define P and Q expressions for bus 2 and 3 using P = Vr*Ir + Vi*Ii, Q = Vi*Ir - Vr*Ii
    model.P2 = Expression(expr=model.v[1] * model.i[1] + model.v[5] * model.i[5])
    model.Q2 = Expression(expr=model.v[5] * model.i[1] - model.v[1] * model.i[5])
    model.P3 = Expression(expr=model.v[2] * model.i[2] + model.v[6] * model.i[6])
    model.Q3 = Expression(expr=model.v[6] * model.i[2] - model.v[2] * model.i[6])

    def objective_rule(m):
        return (m.P2 / P2_target - 1)**2 + (m.Q2 / Q2_target - 1)**2 + (m.P3 / P3_target - 1)**2 + (m.Q3 / Q3_target - 1)**2

    model.obj = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory('ipopt')
    # solver.options['tol'] = 1e-8
    results = solver.solve(model, tee=True)  # tee=True prints solver log

    print("\nSolver termination condition:", results.solver.termination_condition)
    print("Objective value (sum squared mismatch):", value(model.obj))

    v_sol = np.array([value(model.v[i]) for i in range(N)])
    i_sol = M_np.dot(v_sol)
    print("\nVoltage solution vector:", v_sol)
    print("\nCurrent vector (I = M v):", i_sol)

    # Compute P2,Q2,P3,Q3
    P2 = value(model.P2)
    Q2 = value(model.Q2)
    P3 = value(model.P3)
    Q3 = value(model.Q3)

    print(f"P2 computed = {P2:.6f}, target = {P2_target}, err2 = {(P2-P2_target)**2:.6f}")
    print(f"Q2 computed = {Q2:.6f}, target = {Q2_target}, err2 = {(Q2-Q2_target)**2:.6f}")
    print(f"P3 computed = {P3:.6f}, target = {P3_target}, err2 = {(P3-P3_target)**2:.6f}")
    print(f"Q3 computed = {Q3:.6f}, target = {Q3_target}, err2 = {(Q3-Q3_target)**2:.6f}")

if __name__ == "__main__":
    target = (1800, 1700, 1100, 1050)
    solve(target)
