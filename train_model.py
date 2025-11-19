import torch
import torch.nn as nn
import torch.optim as optim
import mat_4bus
from ml_model import VoltagePredictor, PowerFlowLoss

# Initialize model, loss, optimizer
model = VoltagePredictor()
implicit_loss = PowerFlowLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dummy training loop
# Suppose we have 100 samples of (P2,P3,Q2,Q3) 
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

