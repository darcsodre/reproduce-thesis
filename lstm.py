#LSTM (Long Short-Term Memory) é uma rede neural recorrente (RNN), 
#projetada para aprender e lembrar dependências de longo prazo em dados sequenciais.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configurações principais
np.random.seed(42)
torch.manual_seed(42)

# Hiperparâmetros do modelo
num_samples = 2000 # número de amostras (N)
train_batches = 20
val_batches = 2
seq_length = 2000  # comprimento de cada sequência de entrada e saída usada no modelo
hidden_size = 32  # número de unidades ocultas na LSTM
num_layers = 2  # número de camadas LSTM
sigma_v = 0.3
sigma_w = 0.3

# Gerar entrada e saída do sistema
def generate_data(batches, seq_length):
    u = np.random.normal(0, 1, size=(batches, seq_length // 5))
    u = np.repeat(u, 5, axis=1)  # Cada valor mantido por 5 amostras

    v = np.random.normal(0, sigma_v, size=(batches, seq_length))
    w = np.random.normal(0, sigma_w, size=(batches, seq_length))

    y_star = np.zeros((batches, seq_length))
    y = np.zeros((batches, seq_length))

    for i in range(2, seq_length):
        exp_clip = np.clip(y[:, i - 1], -10, 10)
        exp_term1 = 0.8 - 0.5 * np.exp(-exp_clip**2)
        exp_term2 = 0.3 + 0.9 * np.exp(-exp_clip**2)

        y_star[:, i] = (
            exp_term1 * y_star[:, i - 1]
            - exp_term2 * y_star[:, i - 2]
            + u[:, i - 1]
            + 0.2 * u[:, i - 2]
            + 0.1 * u[:, i - 1] * u[:, i - 2]
            + v[:, i]
        )

        y_star[:, i] = np.clip(y_star[:, i], -10, 10)
        y[:, i] = y_star[:, i] + w[:, i]

    return u, y

# Gerar dados de treinamento e validação
u_train, y_train = generate_data(train_batches, seq_length)
u_val, y_val = generate_data(val_batches, seq_length)

# Converter dados para tensores do PyTorch
u_train_tensor = torch.tensor(u_train, dtype=torch.float32).unsqueeze(2)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(2)
u_val_tensor = torch.tensor(u_val, dtype=torch.float32).unsqueeze(2)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(2)

# Definir o modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Criar o modelo
model = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Treinamento
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(u_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    val_outputs = model(u_val_tensor)
    val_loss = loss_fn(val_outputs, y_val_tensor)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Simulação do modelo treinado
model.eval()
u_sim, y_sim = generate_data(1, seq_length)
u_sim_tensor = torch.tensor(u_sim, dtype=torch.float32).unsqueeze(2)

# Selecionar as últimas 100 amostras para plotar
last_samples = 100
sim_outputs = model(u_sim_tensor).detach().numpy().squeeze()
sim_outputs = sim_outputs[-last_samples:]
y_sim = y_sim.squeeze()[-last_samples:]

# Calcular MSE e RMSE
y_real = y_sim
y_pred = sim_outputs
mse = mean_squared_error(y_real, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(range(last_samples), y_sim, label="Real System", linestyle="--")
plt.plot(range(last_samples), sim_outputs, label="LSTM Model")
plt.legend()
plt.title("Comparison of Real System and LSTM Model")
plt.xlabel("Sample")
plt.ylabel("Output")
plt.show()
