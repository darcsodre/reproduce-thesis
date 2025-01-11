#Este código implementa um modelo TCN com as configurações especificadas. 
# Ele treina o modelo nos dados gerados artificialmente, 
# valida o desempenho e compara a simulação do modelo TCN com o sistema real, 
# exibindo um gráfico com 100 amostras. 

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configurações principais
np.random.seed(42)
torch.manual_seed(42)

# Hiperparâmetros do modelo

#tamanho do kernel
kernel_size = 2
#Taxa de dropout para regularizar o modelo.
dropout_rate = 0
#Número de camadas convolucionais no modelo TCN.
num_layers = 5
#Fator de dilatação das convoluções.
dilation = 1
#Número de amostras para o gráfico
num_samples = 100
train_batches = 20
val_batches = 2
seq_length = 100
#v e w ruído gaussiano branco
#ambos com desvios padrão 0,3
sigma_v = 0.3
sigma_w = 0.3

# Gerar entrada e saída do sistema
'''
def generate_data(batches, seq_length):
    u = np.random.normal(0, 1, size=(batches, seq_length // 5))
    u = np.repeat(u, 5, axis=1)  # Cada valor mantido por 5 amostras

    v = np.random.normal(0, sigma_v, size=(batches, seq_length))
    w = np.random.normal(0, sigma_w, size=(batches, seq_length))

    y_star = np.zeros((batches, seq_length))
    y = np.zeros((batches, seq_length))

    for i in range(2, seq_length):
        y_star[:, i] = (
            (0.8 - 0.5 * np.exp(-y[:, i - 1])) * y_star[:, i - 1] ** 2
            - (0.3 + 0.9 * np.exp(-y[:, i - 1])) * y_star[:, i - 2] ** 2
            + u[:, i - 1]
            + 0.2 * u[:, i - 2]
            + 0.1 * u[:, i - 1] * u[:, i - 2]
            + v[:, i]
        )
        y[:, i] = y_star[:, i] + w[:, i]

    return u, y
'''
'''
def generate_data(batches, seq_length):
    u = np.random.normal(0, 1, size=(batches, seq_length // 5))
    u = np.repeat(u, 5, axis=1)  # Cada valor mantido por 5 amostras

    v = np.random.normal(0, sigma_v, size=(batches, seq_length))
    w = np.random.normal(0, sigma_w, size=(batches, seq_length))

    y_star = np.zeros((batches, seq_length))
    y = np.zeros((batches, seq_length))

    for i in range(2, seq_length):
        exp_clip = np.clip(-y[:, i - 1], -100, 100)  # Limitar o argumento da exponencial
        y_star[:, i] = (
            (0.8 - 0.5 * np.exp(exp_clip)) * np.clip(y_star[:, i - 1], -10, 10) ** 2
            - (0.3 + 0.9 * np.exp(exp_clip)) * np.clip(y_star[:, i - 2], -10, 10) ** 2
            + u[:, i - 1]
            + 0.2 * u[:, i - 2]
            + 0.1 * u[:, i - 1] * u[:, i - 2]
            + v[:, i]
        )
        # Limitar os valores de y_star após o cálculo
        y_star[:, i] = np.clip(y_star[:, i], -100, 100)
        y[:, i] = y_star[:, i] + w[:, i]

    return u, y
'''
def generate_data(batches, seq_length):
    u = np.random.normal(0, 1, size=(batches, seq_length // 5))
    u = np.repeat(u, 5, axis=1)  # Cada valor mantido por 5 amostras

    v = np.random.normal(0, sigma_v, size=(batches, seq_length))
    w = np.random.normal(0, sigma_w, size=(batches, seq_length))

    y_star = np.zeros((batches, seq_length))
    y = np.zeros((batches, seq_length))

    for i in range(2, seq_length):
        # Calcular o termo exponencial com clipping adequado
        exp_clip = np.clip(-y[:, i - 1], -10, 10)  # Intervalo ajustado para maior realismo
        exp_term1 = 0.8 - 0.5 * np.exp(exp_clip)
        exp_term2 = 0.3 + 0.9 * np.exp(exp_clip)

        # Atualizar y_star com termos ajustados
        y_star[:, i] = (
            exp_term1 * y_star[:, i - 1] ** 2
            - exp_term2 * y_star[:, i - 2] ** 2
            + u[:, i - 1]
            + 0.2 * u[:, i - 2]
            + 0.1 * u[:, i - 1] * u[:, i - 2]
            + v[:, i]
        )

        # Evitar explosão numérica no resultado de y_star
        y_star[:, i] = np.clip(y_star[:, i], -50, 50)

        # Adicionar o ruído de medição
        y[:, i] = y_star[:, i] + w[:, i]

    return u, y


# Gerar dados de treinamento e validação
u_train, y_train = generate_data(train_batches, seq_length)
u_val, y_val = generate_data(val_batches, seq_length)

# Converter dados para tensores do PyTorch
u_train_tensor = torch.tensor(u_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
u_val_tensor = torch.tensor(u_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Definir o modelo TCN
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else num_channels
            layers.append(
                nn.Conv1d(
                    in_channels,
                    num_channels,
                    kernel_size,
                    padding=(kernel_size - 1),
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(num_channels, output_size, 1)

    def forward(self, x):
        x = self.network(x)
        return self.output_layer(x)

# Criar o modelo
model = TCN(1, 1, 16, kernel_size, dropout_rate)
loss_fn = nn.MSELoss()
#taxa de aprendizado (lr)vou alterar de 0.01 para 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Treinamento
#Almentei o valor de epochs.
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(u_train_tensor)
    outputs = outputs[:, :, -y_train_tensor.size(2):]  # Ajustar tamanho da saída
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    val_outputs = model(u_val_tensor)
    val_outputs = val_outputs[:, :, -y_val_tensor.size(2):]
    val_loss = loss_fn(val_outputs, y_val_tensor)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Simulação do modelo treinado
model.eval()
u_sim, y_sim = generate_data(1, seq_length)
u_sim_tensor = torch.tensor(u_sim, dtype=torch.float32).unsqueeze(1)
y_sim_tensor = torch.tensor(y_sim, dtype=torch.float32).unsqueeze(1)

sim_outputs = model(u_sim_tensor).detach().numpy().squeeze()
sim_outputs = sim_outputs[-num_samples:]
y_sim = y_sim.squeeze()[-num_samples:]

print(y_sim)

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(range(num_samples), y_sim, label="Real System", linestyle="--")
plt.plot(range(num_samples), sim_outputs, label="TCN Model")
plt.legend()
plt.title("Comparison of Real System and TCN Model")
plt.xlabel("Sample")
plt.ylabel("Output")
plt.show()
