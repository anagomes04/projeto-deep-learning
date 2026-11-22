## Processo de execução do projeto

1. **Carregar dados da LUMIERE na pasta `data`**: deve seguir o padrão de nomenclatura da própria LUMIERE (PATIENT-XXX/week-XXX-X).
2. **Dados reais**: `python real_data.py` → gera `processed/real/Patient-*.npz` e `.json`.
3. **Dado sintético (teste)**: `python simulator.py` → gera `processed/sim/test_synthetic.npz` e `.json`.
4. **Treinar**: `python train.py <arquivo-ou-pasta>`, podendo utilizar tanto os dados reais quanto os sintéticos → gera `*_trained.pt` ao lado de cada dataset.
5. **Visualizar**: `python plots.py` → gera PNGs em `figures/` a partir dos CSVs salvos em `results/`.

Os demais arquivos (`pinn.py` e `patient.py`) são utilizados internamente dentro do `real_data.py` e foram criados com intuito de organização de código.