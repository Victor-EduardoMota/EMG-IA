import torch
import torch.nn as nn
import torch.optim as optim
import random

def ruido(valor, nivel=0.05):
    valor += random.uniform(-nivel, nivel)
    return max(0.0, min(1.0, valor))

def gerar_subida(intensidade):
    valores = []
    atual = intensidade * random.uniform(0.2, 0.5)
    for _ in range(4):
        atual += random.uniform(0.05, 0.25)
        atual = min(atual, intensidade)
        valores.append(ruido(atual))

    return valores

def gerar_descida(intensidade):
    valores = []
    atual = intensidade
    for _ in range(4):
        atual -= random.uniform(0.05, 0.25)
        atual = max(0.0 , atual)
        valores.append(ruido(atual))

    return valores
     
def extrair_features(seq):
    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    return seq + diffs
    


entradas = []
saidas = []

for _ in range(500): # Vamos gerar 500 sequências de subida e 500 de descida
    # SUBIDA (contração)
    intensidade = random.uniform(0.3, 1.0)

    dados_subida = gerar_subida(intensidade)

    entradas.append(extrair_features(dados_subida))
    saidas.append(intensidade)  # <-- chave aqui

    intensidade = random.uniform(0.0, 0.4)

    dados_descida = gerar_descida(intensidade)

    entradas.append(extrair_features(dados_descida))
    saidas.append(intensidade * 0.3)

# Converter para Tensores
entradas = torch.tensor(entradas, dtype=torch.float32)
saidas = torch.tensor(saidas, dtype=torch.float32).unsqueeze(1)


#modelo
model = nn.Sequential(

nn.Linear(7, 8),
nn.ReLU(),
nn.Linear(8, 1),
nn.Sigmoid()

)

#treino
criterion = nn.MSELoss()
opttmizer = optim.Adam(model.parameters(), lr=0.01)

for epoca in range(1000):
    output = model(entradas)

    loss = criterion(output, saidas)

    opttmizer.zero_grad()

    loss.backward()

    opttmizer.step()

    if epoca % 100 == 0:
        print(f"Época {epoca}, Loss: {loss.item():.4f}")

#teste
testes = []
for _ in range(5):
    t = gerar_descida(random.uniform(0.5, 1.0))
    testes.append(t)


with torch.no_grad():
    for t in testes:
        entrada = torch.tensor(extrair_features(t), dtype=torch.float32).unsqueeze(0)
        saida = model(entrada)
        valor = saida.item()

        print("Entrada:", t)
        print("Intensidade prevista:", valor)

        
torch.save(model.state_dict(), "modelo_emg.pth")