import torch
import torch.nn as nn
import torch.optim as optim
import mat_4bus
from ml_model import VoltagePredictor, PowerFlowLoss, PreTrain
from timer import Timer
import check_model

timer = Timer(40)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = "cpu"
print('Using device:', DEVICE, str(DEVICE))   

# Initialize model, loss, optimizer
model = VoltagePredictor(DEVICE, nhidden=(10,))
implicit_loss = PowerFlowLoss(DEVICE)
pretrain = PreTrain(DEVICE, [1., 1., 1., 1., 0., 0., 0.])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dummy training loop
# Suppose we have 100 samples of (P2,P3,Q2,Q3) 
stdev = torch.tensor([0.3, 0.3, 0.15, 0.15]).to(DEVICE) / 3
means = torch.tensor([1, 1, 0.6, 0.6]).to(DEVICE)

timer.print("initialized")
print("GPU available:", torch.cuda.is_available())
print("On GPU:", means.device)

#size = [40, 40, 100, 100, 400, 400, 1000, 1000]
size = [100, 300, 1000]
size = [s for s in size for _ in range(4)]
if "cuda" in str(DEVICE):
    size = [s * 100 for s in size]

# Pretrain
X = (torch.randn(15, 200, 4, device=DEVICE) * stdev + means) * 1.80
for epoch in range(200):
    for x in X:
        optimizer.zero_grad()
        V_pred = model(x)
        loss = pretrain(V_pred)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 50 == 0:
        timer.print(f"Pretrain Epoch {epoch}, Loss = {loss.item():.6f}")

print("example")
print(model(x[0]).cpu().detach().numpy().round(3))
    
# Actual train
for i, s in enumerate(size):
    X = (torch.randn(15, s, 4, device=DEVICE) * stdev + means) * 1.80
    
    for epoch in range(200):
        for x in X:
            optimizer.zero_grad()
            V_pred = model(x)
            loss = implicit_loss(V_pred, x)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            timer.print(f"{i} {s} Epoch {epoch}, Loss = {loss.item():.6f}")

# Save trained model
torch.save(model.state_dict(), "voltage_predictor.pt")
check_model.test(model, implicit_loss.GBBG.cpu().numpy())
