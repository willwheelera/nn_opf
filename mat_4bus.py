import numpy as np

def create_Ybus():
    # Define line data (R, X, B)
    nbus = 4
    Ybus = np.zeros((nbus, nbus), dtype=complex)
    lines = np.array([
        [1,   2,   0.01008,   0.0504,   0.1025],
        [1,   3,   0.00744,   0.0372,   0.0775],
        [2,   4,   0.00744,   0.0372,   0.0775],
        [3,   4,   0.01272,   0.0636,   0.1275],
    ])

    # Build Ybus
    for f, t, R, X, B in lines:
        z = R + 1j * X
        y = 1 / z
        Bc = 1j * B / 2  # half-line charging

        f, t = int(f - 1), int(t - 1)
        Ybus[f, f] += y + Bc
        Ybus[t, t] += y + Bc
        Ybus[f, t] -= y
        Ybus[t, f] -= y
    
    return Ybus

def get_GBBG(Y):
    G = Y.real
    B = Y.imag
    return np.block([[G, -B[:, 1:]], [-B[1:], -G[1:, 1:]]]) # no 5th column for V1im


if __name__ == "__main__":
    Ybus = create_Ybus()
    G = Ybus.real
    B = Ybus.imag
    GBBG = get_GBBG(Ybus)
    def test_KCL():
        for V in np.random.random((10, 7, 10)) * 2:
            I_r = G @ V[:4] - B[:, 1:] @ V[4:]
            I_i = B @ V[:4] + G[:, 1:] @ V[4:]
            I = GBBG @ V
            print("check GBBG", np.mean((I_r-I[:4])**2), np.mean((I_i[1:]+I[4:])**2))
            print(I_r.sum(axis=0), I_i.sum(axis=0))

    test_KCL()

# ----------------------------
# Voltage magnitudes and angles (radians)
# ----------------------------
#Vm = np.array([1.06, 1.045, 1.02, 1.01])
#Va_deg = np.array([0, -4.98, -12.72, -10.33])
#Va = np.deg2rad(Va_deg)

