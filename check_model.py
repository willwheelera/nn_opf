import mat_4bus
import torch
import ml_model
import numpy as np

def load():
    Ybus = mat_4bus.create_Ybus()
    GBBG = mat_4bus.get_GBBG(Ybus)
    modelstate = torch.load("voltage_predictor.pt")
    nhidden = [v.size()[0] for k, v in modelstate.items() if "weight" in k]
    nhidden = nhidden[:-1]
    print("nhidden", nhidden)
    model = ml_model.VoltagePredictor("cpu", nhidden=nhidden)
    model.load_state_dict(modelstate)
    return model, GBBG

def test(model, GBBG):
    target = torch.tensor((1.800, 1.700, 1.100, 1.050), device="cpu") 
    V_pred = model(target).to("cpu").detach().numpy()
    V = np.zeros(8)
    V[[0,1,2,3,5,6,7]] = V_pred
    print("V", V.round(3))
    print("I", (GBBG @ V).round(3))
    S = mat_4bus.solve_powers(V, GBBG)
    print(target)
    print("err", (S[[1, 2, 5, 6]] - target.detach().numpy()).round(3))

if __name__ == "__main__":
    test(*load())
