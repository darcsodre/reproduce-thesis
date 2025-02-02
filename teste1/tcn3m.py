#Este código implementa um modelo TCN com as configurações especificadas. 
# Ele treina o modelo nos dados gerados artificialmente, 
# valida o desempenho e compara a simulação do modelo TCN com o sistema real, 
# exibindo um gráfico com 100 amostras. 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configurações principais
np.random.seed(42)
torch.manual_seed(42)

# Hiperparâmetros do modelo
kernel_size = 2 #Tamanho do kernel
dropout_rate = 0 #Nenhum neurônio será desativado
num_layers = 5 #Nº de camadas convolucionais no modelo TCN.
dilation = 1 #Fator de dilatação das convoluções.

#Configurações do conjunto de dados
num_samples = 100   #Nº total de amostras usadas para o gráfico gerado.
train_batches = 20   #Nº de batches usados no conjunto de treinamento.
val_batches = 2      #Nº de batches no conjunto de validação. 
seq_length = 100     #comp. de cada sequência de entrada e saída usada no modelo do sistema real.

#Parâmetros de ruído
sigma_v = 0.3 #Desvio padrão do ruído de processo.
sigma_w = 0.3 #Desvio padrão do ruído de medição. 

# Gerar entrada e saída do sistema

def generate_data(batches, seq_length):
    # Geração da entrada:
    u = np.random.normal(0, 1, size=(batches, seq_length // 5))
    u = np.repeat(u, 5, axis=1)  # Cada valor mantido por 5 amostras
    
    #Gerando ruídos v e w:
    v = np.random.normal(0, sigma_v, size=(batches, seq_length))
    w = np.random.normal(0, sigma_w, size=(batches, seq_length))

    #Inicialização de matrizes para saidas:
    y_star = np.zeros((batches, seq_length)) #saida intermediária do sistema SEM ruído de medição.
    y = np.zeros((batches, seq_length)) #saida final do sistema, COM ruído de medição.

    for i in range(2, seq_length):
        # Calcular o termo exponencial com clipping adequado
        exp_clip = np.clip(y[:, i - 1], -10, 10)  # Intervalo ajustado para maior realismo
        exp_term1 = 0.8 - 0.5 * np.exp(-exp_clip**2) #ajuste da funcao 
        exp_term2 = 0.3 + 0.9 * np.exp(-exp_clip**2) #ajuste da funcao 

        # Atualizar y_star com termos ajustados
        y_star[:, i] = (
            exp_term1 * y_star[:, i - 1]
            - exp_term2 * y_star[:, i - 2]
            + u[:, i - 1]
            + 0.2 * u[:, i - 2]
            + 0.1 * u[:, i - 1] * u[:, i - 2]
            + v[:, i]
        )

        # Evitar explosão numérica no resultado de y_star
        y_star[:, i] = np.clip(y_star[:, i], -10, 10)

        # Adicionar o ruído de medição
        y[:, i] = y_star[:, i] + w[:, i]

    return u, y

#Pre-processamento 
# Gerar dados de treinamento e validação
u_train, y_train = generate_data(train_batches, seq_length) #funcao e parametros.Dados de entrada de saida p/cj de treinamento.
u_val, y_val = generate_data(val_batches, seq_length) # Dados de entrada e saida p/ cj de validaçao.

# Converter os dados gerados pelo generate_data para tensores do PyTorch. 
u_train_tensor = torch.tensor(u_train, dtype=torch.float32).unsqueeze(1) #tensor de entrada para o treinamento
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) #tensor de saida para treinamento
u_val_tensor = torch.tensor(u_val, dtype=torch.float32).unsqueeze(1) #tesor de entrada para validacao
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1) #tensor de saida 

# Definir o modelo TCN
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else num_channels
            layers.append(
                nn.Conv1d(
                    in_channels, #ajusta para conexão correta entre as camadas
                    num_channels, #tbm
                    kernel_size,
                    padding=(kernel_size - 1),#saida tem o mesmo compr.de entrada
                    dilation=dilation,  #controla o espaçamento entre valores usando o calculo do filtro
                )
            )
            layers.append(nn.ReLU())       
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(num_channels, output_size, 1)

    def forward(self, x): # Define como os dados percorrem o modelo durante a propagação.
        x = self.network(x)
        return self.output_layer(x)

# Criar o modelo
model = TCN(1, 1, 16, kernel_size, dropout_rate)
loss_fn = nn.MSELoss() #funçao de perda.Calcula o erro médio quadrático entre as previsões e os valores reais.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #Atualiza os pesos do modelo.

# Treinamento
#Aumentei o valor de epochs.
epochs = 500
for epoch in range(epochs):
    model.train() #Coloca o modelo no modo de treinamento.
    optimizer.zero_grad() #Zera os gradientes acumulados das iterações anteriores.
    outputs = model(u_train_tensor) #Passa os dados de entrada pelo modelo para obter as previsões.
    
    #print(-y_train_tensor.size(2))
    outputs = outputs[:, :, -y_train_tensor.size(2):]  # Ajustar tamanho da saída
    loss = loss_fn(outputs, y_train_tensor) #Calcula a perda entre as previsões e os valores reais.
    loss.backward() #Calcula os gradientes dos pesos
    optimizer.step() #Atualiza os pesos com base nos gradientes.

    #Validação:
    model.eval() #Coloca o modelo no modo de avaliação (sem dropout, etc.).
    val_outputs = model(u_val_tensor) #
    val_outputs = val_outputs[:, :, -y_val_tensor.size(2):]
    val_loss = loss_fn(val_outputs, y_val_tensor)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Simulação do modelo treinado
model.eval()
u_sim, y_sim = generate_data(1, seq_length)
u_sim_tensor = torch.tensor(u_sim, dtype=torch.float32).unsqueeze(1)
y_sim_tensor = torch.tensor(y_sim, dtype=torch.float32).unsqueeze(1)

# Selecionar as últimas 100 amostras para plotar
last_samples = 100
sim_outputs = model(u_sim_tensor).detach().numpy().squeeze()
sim_outputs = sim_outputs[-last_samples:]  # Pegar as últimas 100 amostras
y_sim = y_sim.squeeze()[-last_samples:]    # Pegar as últimas 100 amostras

#print(y_sim)

# Gerar as previsões e os valores reais
y_real = y_sim  # Valores reais gerados pelo sistema
y_pred = sim_outputs  # Valores previstos pelo modelo

# Calcular MSE
mse = mean_squared_error(y_real, y_pred)

# Calcular RMSE
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.4f}")

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(range(last_samples), y_sim, label="Real System", linestyle="--")
plt.plot(range(last_samples), sim_outputs, label="TCN Model")
plt.legend()
plt.title("Comparison of Real System and TCN Model")
plt.xlabel("Sample")
plt.ylabel("Output")
plt.show()
