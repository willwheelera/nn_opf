import torch
import torch.nn as nn
import torch.optim as optim
import mat_4bus

# -------------------------
# Define neural network
# -------------------------
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


# -------------------------
# Define implicit "physics" layer (custom loss)
# -------------------------
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
        Ic = (V @ GBBG)
        S = torch.zeros(V.shape)
        S[:, :4] = Ic[:, :4] * V[:, :4]
        S[:, 1:4] -= Ic[:, 4:] * V[:, 4:]
        S[:, 4:] = Ic[:, 1:4] * V[:, 4:] + Ic[:, 4:] * V[:, 1:4]
        err = torch.mean((S[:, [1, 2, 5, 6]] / targets - 1)**2)
        cost = COST * torch.mean(S[:, 0] + S[:, 3])
        return err + cost
        


# -------------------------
# Initialize model, loss, optimizer
# -------------------------
model = VoltagePredictor()
implicit_loss = PowerFlowLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Dummy training loop
# -------------------------
# Suppose we have 100 samples of (P2,P3,Q2,Q3) 
#X3 = torch.randn(10, 900, 4) * torch.tensor([30, 30, 0.05, 0.05]) + torch.tensor([180, 180, 0.6, 0.6])   # inputs
#X2 = torch.randn(10, 300, 4) * torch.tensor([30, 30, 0.05, 0.05]) + torch.tensor([180, 180, 0.6, 0.6])   # inputs
#X1 = torch.randn(10, 100, 4) * torch.tensor([30, 30, 0.05, 0.05]) + torch.tensor([180, 180, 0.6, 0.6])   # inputs
#X0 = torch.randn(10, 40, 4) * torch.tensor([30, 30, 0.05, 0.05]) + torch.tensor([180, 180, 0.6, 0.6])   # inputs
stdev = torch.tensor([0.2, 0.2, 0.05, 0.05]) 
means = torch.tensor([1, 1, 0.6, 0.6])

size = [40, 40, 100, 100, 400, 400, 1000, 1000]
for i, s in enumerate(size):
    X = (torch.randn(10, s, 4) * stdev + means) * 180
    
    for epoch in range(300):
        for x in X:
                optimizer.zero_grad()
                
                V_pred = model(x)
                loss = implicit_loss(V_pred, x)
                
                loss.backward()
                optimizer.step()

        if epoch % 100 == 0:
            print(f"{i} ({s}) Epoch {epoch}, Loss = {loss.item():.6f}")

# Save trained model
torch.save(model.state_dict(), "voltage_predictor.pt")

