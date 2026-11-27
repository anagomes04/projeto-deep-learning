import math

import torch 
from torch import nn
from tqdm import tqdm

from .neural_network import NeuralNetwork

class IPINN:

    def __init__(self, model: NeuralNetwork, d_init: float = 1e-2):

        self.model = model 
        self.log_d = nn.Parameter(torch.tensor(math.log(d_init)))

    def data_loss(self, data_input: torch.Tensor, data_target: torch.Tensor):

        return torch.mean((self.model(data_input) - data_target)**2)

    def residual_loss(self, collocation_points: torch.Tensor):

        # Ordem das colunas de entrada: 0 = t, 1 = x, 2 = y
        input = collocation_points.clone().requires_grad_(True)
        u = self.model(input)

        grads = torch.autograd.grad(u, input, torch.ones_like(u), create_graph=True)[0]
        u_t, u_x, u_y = grads[:, 0], grads[:, 1], grads[:, 2]

        # CORRECAO: segundas derivadas puras (x -> coluna 1, y -> coluna 2).
        # Antes: colunas 0 e 1, que davam derivadas mistas (u_xt e u_yx).
        u_xx = torch.autograd.grad(u_x, input, torch.ones_like(u_x), create_graph=True)[0][:, 1]
        u_yy = torch.autograd.grad(u_y, input, torch.ones_like(u_y), create_graph=True)[0][:, 2]
        
        D = torch.exp(self.log_d)
        residual = u_t - D * (u_xx + u_yy) - u.squeeze(-1)
        return torch.mean(residual**2)

    def fit(self, trainning_data: dict, epochs: int = 1_000, learning_rates: list[float] = [1e-3, 1e-2], device: str = 'cpu',
            seed: int | None = None, resample_fn=None, resample_every: int = 0):
        """
        seed: se informado, fixa a semente (reprodutibilidade entre repeticoes).
        resample_fn: funcao opcional sem argumentos que devolve novos pontos de colocacao
                     (tensor N x 3); usada a cada `resample_every` epocas (0 = desligado).
        """

        if seed is not None:
            torch.manual_seed(seed)
            
        self.model = self.model.to(device)
        self.log_d = nn.Parameter(self.log_d.detach().to(device))

        optim_model = torch.optim.AdamW(self.model.parameters(), lr = learning_rates[0], weight_decay = 0.0)
        optim_param = torch.optim.Adam([self.log_d], lr = learning_rates[1])
        scheduler_param = torch.optim.lr_scheduler.CosineAnnealingLR(optim_param, T_max = epochs, eta_min = 1e-6)
        scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, T_max = epochs, eta_min = 1e-6)

        data_input_obs = trainning_data['experimental_input'].to(device)
        data_target_obs = trainning_data['experimental_target'].to(device)
        data_input_ic = trainning_data['ic_input'].to(device)
        data_target_ic = trainning_data['ic_target'].to(device)
        collocation_data = trainning_data['collocation_data'].to(device)

        history = {'loss': [], 'data_loss': [], 'residual_loss': [], 'ic_loss': [], 'dl': []}
        for epoch in tqdm(range(epochs)):

            if resample_fn is not None and resample_every > 0 and epoch > 0 and epoch % resample_every == 0:
                collocation_data = resample_fn().to(device)

            optim_model.zero_grad()
            optim_param.zero_grad()
            
            data_loss_value = self.data_loss(data_input_obs, data_target_obs)
            ic_loss_value = self.data_loss(data_input_ic, data_target_ic)
            residual_loss_value = self.residual_loss(collocation_data)
            loss = data_loss_value + ic_loss_value + residual_loss_value
            
            loss.backward()
            optim_model.step()
            optim_param.step()
            scheduler_param.step()
            scheduler_model.step()

            history['loss'].append(loss.item())
            history['data_loss'].append(data_loss_value.item())
            history['residual_loss'].append(residual_loss_value.item())
            history['ic_loss'].append(ic_loss_value.item())
            history['dl'].append(torch.exp(self.log_d).item())

        return history