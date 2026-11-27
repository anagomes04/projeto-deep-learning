import sys
import os

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if raiz_projeto not in sys.path:
    sys.path.append(raiz_projeto)

import torch 
import numpy as np 
import time

from torch import nn
from src.neural_network import NeuralNetwork
from src.physics_informed_model import IPINN

proporcao = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
collocation_points_size = 1_000
epocas = [20_000]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Usando dispositivo: {device}')


if __name__ == '__main__':

    for i in range(1, 11):

        print(f'Iniciando treinamento {i:02d}')
        local = f'dados/experimento_01/treinamento_{i:02d}'

        # Criando diretórios necessários
        os.makedirs(local, exist_ok = True)
        os.makedirs(local + '/historico', exist_ok = True)
        os.makedirs(local + '/modelos', exist_ok = True)

        # Carregando os dados simulados por diferenças finitas
        dados = np.load('dados/numericos/data_002.npz')
        input = dados['input']
        target = dados['target']
        param = dados['parameters']
        data_total_size = input.shape[0]

        for ep in epocas:

            for prop in proporcao:

                model = NeuralNetwork([3] + 3 * [256] + [1], torch.tanh)
                inverse_pinn = IPINN(model)

                quantidade_dados = round(prop * data_total_size)

                # Filtrando uma amostra aleatória de 1.000 dados experimentais para treinamento
                idx = np.random.choice(range(data_total_size), size = quantidade_dados, replace = False)

                # Convertendo os dados para tensores do PyTorch
                data_input_obs = torch.from_numpy(input[idx]).float().to(device)
                data_target_obs = torch.from_numpy(target[idx]).float().to(device)

                # Definindo dados para condição inicial conhecida
                data_input_ic = torch.from_numpy(input[:2500]).float().to(device)
                data_target_ic = torch.from_numpy(target[:2500]).float().to(device)

                # Definindo os pontos de colocation para o cálculo do resíduo da equação diferencial residual
                x = torch.FloatTensor(collocation_points_size).uniform_(0, 1)
                y = torch.FloatTensor(collocation_points_size).uniform_(0, 1)
                t = torch.FloatTensor(collocation_points_size).uniform_(0, 1)

                collocation_data = torch.vstack([t, x, y]).T.requires_grad_(True).to(device)

                # Agrupando dados de treinamento
                trainning_data = {
                    'experimental_input': data_input_obs,
                    'experimental_target': data_target_obs,
                    'ic_input': data_input_ic,
                    'ic_target': data_target_ic,
                    'collocation_data': collocation_data
                }

                inicio = time.perf_counter()
                history = inverse_pinn.fit(
                    trainning_data,
                    ep,
                    device = device
                )
                fim = time.perf_counter()

                # Verificando proporção dos dados utilizada
                print(f'Proporção dos Dados: {prop:.2%}')

                # Verificando tempo de treinamento
                h, resto = divmod(int(fim - inicio), 3600)
                m, s = divmod(resto, 60)
                print(f'Tempo de Treinamento: {h}h {m}m {s}seg')

                # Verificando precisão da inferência de D*
                dl_estimado = torch.exp(inverse_pinn.log_d).item()
                dl_ref = (param[0] / (param[1] * param[2] ** 2)).item()
                erro = np.linalg.norm(dl_estimado - dl_ref) / np.linalg.norm(dl_ref)
                print(f'Erro: {erro:.2%}    Estimado: {dl_estimado:.7f}     Referência: {dl_ref:.7f}')
                
                print('-' * 50)

                # salvando modelo e histórico de treinamento
                model.save(local + f'/modelos/model_{int(ep)}_{int(prop * 100)}.pth')
                np.save(local + f'/historico/historico_{int(ep)}_{int(prop * 100)}.npy', history)

                with open(local + f'/metadados_treino_{i:02d}.txt', 'a') as arq:

                    arq.write(f'Quantidade Pontos de Colocacao: {collocation_points_size}\n')
                    arq.write(f'Quantidade Epocas: {ep}\n')
                    arq.write(f'Proporcao dos Dados Experimentais: {prop:.2%}\n')
                    arq.write(f'Tempo de Treinamento: {h}h {m}m {s}seg\n')
                    arq.write(f'Erro: {erro:.2%}    Estimado: {dl_estimado:.7f}     Referencia: {dl_ref:.7f}\n')
                    arq.write('-' * 50 + '\n')

            print('\n\n')