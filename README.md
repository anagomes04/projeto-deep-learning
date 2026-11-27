## Processo de execução do projeto

Esse projeto é configurado em quatro diretórios:
- **modelo_reativo_difusivo**: diretório criado para implementação considerando modelo reativo difusivo clássico.
- **modelo_adimensionalizado**: diretório criado para implementação considerando modelo reativo difusivo adimensionalizado.
- **captacao_temporal**: diretório criado para implementação considerando abordagem adimensionalizada com tempo fixo normalizado.
- **modelo_fisher_kolmogorov**: diretório criado para implementação considerando modelo de Fisher-Kolmogorov.

Para a execução do modelo de Fisher-Kolmogorov, seguir os seguintes passos:
1. **Carregar dados da LUMIERE na pasta `data`**: deve seguir o padrão de nomenclatura da própria LUMIERE (PATIENT-XXX/week-XXX-X).
2. **Dados reais**: `python real_data.py` → gera `processed/real/Patient-*.npz` e `.json`.
3. **Dado sintético (teste)**: `python simulator.py` → gera `processed/sim/test_synthetic.npz` e `.json`.
4. **Treinar**: `python train.py <arquivo-ou-pasta>`, podendo utilizar tanto os dados reais quanto os sintéticos → gera `*_trained.pt` ao lado de cada dataset.
5. **Visualizar**: `python plots.py` → gera PNGs em `figures/` a partir dos CSVs salvos em `results/`.

Os demais arquivos (`pinn.py` e `patient.py`) são utilizados internamente dentro do `real_data.py` e foram criados com intuito de organização de código.

Para os demais, seguir a estrutura abaixo:
- **Notebooks**: onde é localização o arquivo **main.ipynb** responsável por apresentar as principais features da implementação e plotagens.
- **Dados**: onde ficam localizados os dados sintéticos e experimentos (estes possuem uma pasta **Scripts** no diretório para a automação dos experimentos);

