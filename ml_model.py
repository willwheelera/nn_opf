import torch
import torch.nn as nn
import mat_4bus

# Define neural network
class VoltagePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),   # inputs: P2, P3, Q2, Q3
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 7)    # outputs: V1re, V2re, V2im, V3re, V3im, V4re, V4im
        )

    def forward(self, x):
        return self.net(x)


# Define implicit "physics" layer (custom loss)
Y = mat_4bus.create_Ybus()
GBBG = torch.tensor(mat_4bus.get_GBBG(Y), dtype=torch.float32)
COST = 0.#1e-2
class PowerFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.Y maps inp voltage to current.conj (7, 7)

    def forward(self, V, targets=None):
        """
        V: [batch, 7] predicted voltages
        targets: optional true data or desired values for training
        """
        Ic = (V @ GBBG) # conjugate
        S = torch.zeros(V.shape)
        Vr, Vi = V[:, 1:4], V[:, 4:]
        Ir, Ii = Ic[:, 1:4], Ic[:, 4:]
        S[:, 0] = Ic[:, 0] * V[:, 0] # slack bus Vi=0
        S[:, 1:4] = Ir * Vr - Ii * Vi
        S[:, 4:7] = Ir * Vi + Ii * Vr
        err = torch.mean((S[:, [1, 2, 5, 6]] / targets - 1)**2)
        cost = COST * torch.mean(S[:, 0] + S[:, 3])
        return err + cost
        
