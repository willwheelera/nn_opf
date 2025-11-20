import torch
import torch.nn as nn
import mat_4bus

# Define neural network
class VoltagePredictor(nn.Module):
    def __init__(self, device, nhidden=(32,)):
        super().__init__()
        layers = []
        prev = 4 # inputs: P2, P3, Q2, Q3
        for nh in nhidden:
            layers.append(nn.Linear(prev, nh, device=device))
            layers.append(nn.Tanh())
            prev = nh
        # outputs: V1re, V2re, V2im, V3re, V3im, V4re, V4im
        layers.append(nn.Linear(prev, 7, device=device))
        self.net = nn.Sequential(*layers)
        self.device = device
        
    def forward(self, x):
        return self.net(x)


# Define implicit "physics" layer (custom loss)
class PowerFlowLoss(nn.Module):
    def __init__(self, device, cost=0.):
        super().__init__()
        # self.Y maps inp voltage to current.conj (7, 7)
        Y = mat_4bus.create_Ybus()
        self.GBBG = torch.tensor(mat_4bus.get_GBBG(Y), dtype=torch.float32, device=device)
        self.cost = cost
        self.device = device

    def forward(self, V, targets=None):
        """
        V: [batch, 7] predicted voltages
        targets: optional true data or desired values for training
        """
        Ic = (V @ self.GBBG[[0,1,2,3,5,6,7]]) # conjugate
        S = torch.zeros((V.shape[0], 8), device=self.device)
        Vi = torch.zeros((V.shape[0], 4), device=self.device)
        Vr, Vi[:, 1:4] = V[:, :4], V[:, 4:]
        Ir, Ii = Ic[:, :4], Ic[:, 4:]
        #S[:, 0] = Ic[:, 0] * V[:, 0] # slack bus Vi=0
        S[:, 0:4] = Ir * Vr - Ii * Vi
        S[:, 4:8] = Ir * Vi + Ii * Vr
        err = torch.mean((S[:, [1, 2, 5, 6]] / targets - 1)**2)
        cost = self.cost * torch.mean(S[:, 0] + S[:, 3])
        return err + cost
        
class PreTrain(nn.Module):
    def __init__(self, device, V_init):
        super().__init__()
        self.device = device
        self.V_init = torch.tensor(V_init, device=device)

    def forward(self, V, targets=None):
        return torch.mean((V - self.V_init)**2)
    
