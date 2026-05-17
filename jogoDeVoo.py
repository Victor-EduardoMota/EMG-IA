
import pygame
import random
import torch
import torch.nn as nn

neutro = 0.5  # valor inicial

calibrando = True
contador = 0
soma = 0

largura = 1366
altura = 700      

tile = 150     
x = 100
y = 100

moedasx = largura
moedasy = random.randint(0, altura - 30 )

tirox = largura
tiroy = random.randint(0, altura - 80 )

velocidade_tiro = 30

fly = 5
gravity = 1.0
velocidade_y = 0
pygame.init()
tela = pygame.display.set_mode((largura, altura))
clock = pygame.time.Clock()



model = nn.Sequential(
    nn.Linear(7,8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load("modelo_emg.pth"))
model.eval()

def extrair_features(seq):
    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    return seq + diffs

sequencia = [0.0, 0.0, 0.0, 0.0]
forca_suave = 0.0

def pontos(moedas, moedasx, moedasy, x, y):
    player = pygame.Rect(x, y, 50, 50)
    shoot = pygame.Rect(moedasx, moedasy, 50, 60)
    
    if player.colliderect(shoot):
        moedas += 1
        print(moedas)
        moedasx = largura
        moedasy = random.randint(0, altura - 60)



    return moedas, moedasx, moedasy
    

def morte(tirox, tiroy, x, y):
    player = pygame.Rect(x, y, 50, 50)
    tiro = pygame.Rect(tirox, tiroy, 50, tile)
    
    if player.colliderect(tiro):
        return False
    else:
        return True

def mover_moeda(moedasx, moedasy):
    moedasx -= velocidade_tiro

    if moedasx < 0:
        moedasx = largura
        moedasy = random.randint(0, altura - 60)

    return moedasx, moedasy

def mover_tiro(tirox, tiroy):
    tirox -= velocidade_tiro

    if tirox < 0:
        tirox = largura
        tiroy = random.randint(0, altura - tile)

    return tirox, tiroy    



rodando = True
moedas =0 

while rodando:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            rodando = False
    
    novo_valor = sequencia[-1] + random.uniform(-0.03, 0.07)
    novo_valor = max(0.0, min(1.0, novo_valor))

    sequencia.pop(0)
    sequencia.append(novo_valor)

    # --- IA ---
    entrada = torch.tensor(extrair_features(sequencia), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        forca = model(entrada).item()

    # --- SUAVIZAÇÃO ---
    
    
# suaviza primeiro
    forca_suave = 0.9 * forca_suave + 0.1 * forca

    #calibra com valor REAL (ou suavizado, mas consistente)
    if calibrando:
        soma += forca
        contador += 1

    if contador > 60:
        neutro = soma / contador
        calibrando = False
        print("Neutro calibrado:", neutro)
    

    forca_normalizada = (forca_suave - neutro) * 10


    forca_normalizada = max(-1, min(1, forca_normalizada))
    if abs(forca_normalizada) < 0.07:
        forca_normalizada = 0
    velocidade_y += gravity
    velocidade_y -= forca_normalizada * fly
    velocidade_y *= 0.9
    y += velocidade_y
    print("F:", round(forca_suave,2), "N:", round(neutro,2), "FN:", round(forca_normalizada,2))
    
    if y < 0:
        y = 0
        velocidade_y = 0

    if y > altura - 50:
        y = altura - 50
        velocidade_y = 0
    
    
    tirox, tiroy = mover_tiro(tirox, tiroy)
    moedas, moedasx, moedasy = pontos(moedas, moedasx, moedasy, x, y)
    moedasx, moedasy = mover_moeda(moedasx, moedasy)
    #rodando = morte (tirox, tiroy, x,y)
    
    media = sum(sequencia) / len(sequencia)
    sequencia = [v - media + 0.5 for v in sequencia]

    tela.fill((0,0,0))
    pygame.draw.rect(tela, (0,0,255), (x, y, 50, 50))
    pygame.draw.rect(tela, (255,255,0), (moedasx, moedasy, 50, 60))
    pygame.draw.rect(tela, (255,0   ,0), (tirox, tiroy, 50, tile))

    
    pygame.display.update()


