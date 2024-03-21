import json
import joblib
import math
import numpy as np

def ler_dados_animais(caminho_arquivo):
    dados_animais = []
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        for linha in linhas:
            campos = linha.strip().split(';')
            animal = {
                'id': campos[0],
                'nome': campos[1],
                'genero': campos[2],
                'raca': campos[3],
            }
            dados_animais.append(animal)
    return dados_animais

def salvar_dados_json(caminho_json, animais):
    # Salvar dados em arquivo JSON
    with open(caminho_json, 'w') as json_file:
        json.dump(animais, json_file, indent=4)

    print(f'Dados salvos em {caminho_json}')



def calcular_fator_normalizacao(animais, lista_score, genero, raca):
    soma_produtos = 0.0
    for score in lista_score:
        score_id = score[1]
        for animal in animais:
            if animal['nome'] == score[0]:
                g = animal['genero']
                b = animal['raca']
                break
        ind_g = 1 if g == genero else 0
        ind_b = 1 if b == raca else 0
        produto = score_id * ind_b * ind_g
        soma_produtos += produto
    return soma_produtos


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calcular_score_modificado(genero, raca, score_identificacao, animais):
    fator_normalizacao = calcular_fator_normalizacao(animais, score_identificacao, genero, raca)
    score_modificado = []
    for score in score_identificacao:
        score_id = score[1]
        for animal in animais:
            if animal['nome'] == score[0]:
                g = animal['genero']
                b = animal['raca']
                break
        ind_g = sigmoid(1 if g == genero else 0)
        ind_b = sigmoid(1 if b == raca else 0)
        # Calcular o score modificado usando a fórmula modificada
        score_modificado.append((score_id * ind_g * ind_b) / fator_normalizacao)

    return score_modificado


def normalizar_e_inverter_distancias(dados, fator_escala=0.2):
    # Extrair as distâncias para normalização
    distancias = np.array([dist for _, dist in dados])

    # Normalizar as distâncias com ajuste do fator de escala
    distancias_normalizadas = 1 - fator_escala * (distancias - distancias.min()) / (distancias.max() - distancias.min())

    # Atualizar a lista de dados com as distâncias normalizadas e invertidas
    dados_normalizados = [(rotulo, dist_normalizada) for (rotulo, _), dist_normalizada in zip(dados, distancias_normalizadas)]

    return dados_normalizados

# Exemplo de uso
caminho_arquivo = 'D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/Datasets/dogs-dados.txt'
animais = ler_dados_animais(caminho_arquivo)
# print(animais)
# caminho_json = 'D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/Datasets/dados_animais.json'
# salvar_dados_json(caminho_json, animais)
cachorros_ordenados_euclidiana = joblib.load("D:/Bah/Documentos/ESTUDO/UFRR/TCC/TCC-Codes/DogIdentificationCNN/artefacts/cachorros_ordenados_euclidiana.pkl")
print(cachorros_ordenados_euclidiana)
print("############")
normalizado = normalizar_e_inverter_distancias(cachorros_ordenados_euclidiana)
print(normalizado)
print("############")
scores = calcular_score_modificado("m", "srd", normalizado, animais)
cachorros_scores_novos = [(cachorro[0], novo_score) for cachorro, novo_score in zip(cachorros_ordenados_euclidiana, scores)]
cachorros_scores_novos.sort(key=lambda x: x[1], reverse=True)  # Ordena pelo novo score

print("Cachorros Ordenados com Novos Scores:")
print(cachorros_scores_novos)

