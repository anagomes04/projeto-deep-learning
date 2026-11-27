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

collocation_points_size = [1_000, 5_000, 10_000, 15_000, 20_000]
epocas = 15_000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Usando dispositivo: {device}')


if __name__ == '__main__':

    for i in range(8, 11):

        print(f'Iniciando treinamento {i:02d}')
        local = f'dados/experimento_02/treinamento_{i:02d}'

        # Criando diretórios necessários
        os.makedirs(local, exist_ok = True)
        os.makedirs(local + '/historico', exist_ok = True)
        os.makedirs(local + '/modelos', exist_ok = True)

        # Carregando os dados simulados por diferenças finitas
        dados = np.load('dados/numericos/data_002.npz')
        input = dados['input']
        target = dados['target']
        param = dados['parameters']
        #data_total_size = input.shape[0]


        for pc in collocation_points_size:

            model = NeuralNetwork([3] + 3 * [256] + [1], torch.tanh)
            inverse_pinn = IPINN(model)

            # Convertendo os dados para tensores do PyTorch
            data_input_obs = torch.from_numpy(input).float().to(device)
            data_target_obs = torch.from_numpy(target).float().to(device)

            # Definindo dados para condição inicial conhecida
            data_input_ic = torch.from_numpy(input[:2500]).float().to(device)
            data_target_ic = torch.from_numpy(target[:2500]).float().to(device)

            # Definindo os pontos de colocation para o cálculo do resíduo da equação diferencial residual
            x = torch.FloatTensor(pc).uniform_(0, 1)
            y = torch.FloatTensor(pc).uniform_(0, 1)
            t = torch.FloatTensor(pc).uniform_(0, 1)
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
                epocas,
                device = device
            )
            fim = time.perf_counter()

            # Verificando proporção dos dados utilizada
            print(f'Proporção dos Dados: {1.:.2%}')

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
            model.save(local + f'/modelos/model_cp_{int(pc)}.pth')
            np.save(local + f'/historico/historico_cp_{int(pc)}.npy', history)

            with open(local + f'/metadados_treino_{i:02d}.txt', 'a') as arq:

                arq.write(f'Quantidade Pontos de Colocacao: {pc}\n')
                arq.write(f'Quantidade Epocas: {epocas}\n')
                arq.write(f'Proporcao dos Dados Experimentais: {1.:.2%}\n')
                arq.write(f'Tempo de Treinamento: {h}h {m}m {s}seg\n')
                arq.write(f'Erro: {erro:.2%}    Estimado: {dl_estimado:.7f}     Referencia: {dl_ref:.7f}\n')
                arq.write('-' * 50 + '\n')

        print('\n\n')