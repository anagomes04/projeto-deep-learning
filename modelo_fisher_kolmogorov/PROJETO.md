# Projeto PINN - Crescimento Tumoral 3D (Fisher-Kolmogorov)

Este projeto treina uma PINN (Physics-Informed Neural Network) para estimar os parâmetros
de difusão (D) e proliferação (rho) do tumor a partir de MRIs segmentadas, usando a
equação de Fisher-Kolmogorov 3D:

```
du/dt = D * laplacian(u) + rho * u * (1 - u)
```

Fluxo geral do pipeline:

```
data/Patient-*/week-* ──► main.py ──► processed/pinn/*.npz + *.json
                                              │
                        simulator.py ──► data_sim/test_synthetic.npz + .json
                                              │
                                              ▼
                                           train.py ──► *_trained.pt
                                              │
                                              ▼
                                           plots.py ──► figures/*.png
```

---

## 1. `patient.py` — Estrutura de dados do paciente e sistema de coordenadas

Define duas classes utilitárias usadas por `main.py`:

### `PatientPath`
- Encapsula um diretório de paciente.
- No construtor, varre o diretório em busca das pastas `week-*` (cada semana é um ponto no tempo) e armazena em ordem cronológica.

### `CoordinateFrame`
Converte entre três sistemas de coordenadas: índice de voxel, posição em mm (espaço anatômico RAS) e coordenada normalizada em `[-1, 1]` (entrada da PINN).

Passo a passo do construtor:
1. Recebe a matriz `affine` do NIfTI e a máscara `brain` (booleana).
2. Pega todos os voxels do cérebro.
3. Converte o canto mínimo e máximo da bounding box para mm.
4. `center` = ponto médio da caixa.
5. `scale` = metade do maior lado da caixa (igual nos três eixos para preservar forma e laplaciano).

Métodos:
- `vox_to_mm(ijk)`: multiplica índices pela parte linear da affine e soma a translação.
- `vox_to_norm(ijk)`: `(mm - center) / scale` → entrada que a rede recebe.

---

## 2. `pinn.py` — Amostragem dos pontos usados na PINN

Define a configuração `PINNDataConfig` e três funções de amostragem que geram as nuvens de pontos usadas pela rede.

### Convenção de rótulos
- `HEALTHY = 0`, `EDEMA = 1`, `CORE = 2`.
- Cada rótulo tem um intervalo esperado de densidade `u`: healthy `[0, 0.25]`, edema `[0.25, 0.75]`, core `[0.75, 1.0]`.

### `build_class_map(seg, brain, cfg)`
Converte os rótulos originais da segmentação DeepBraTumIA (1 = realce, 2 = necrose, 3 = edema) para as classes da PINN; voxels fora do cérebro recebem `-1` (ignorar).

### `sample_voxels(mask, n, rng, near=None, near_fraction=0)`
Sorteia `n` voxels sem repetição dentro de `mask`. Se `near` for fornecida, faz amostragem estratificada: uma fração `near_fraction` vem de dentro de `near` (perto do tumor) e o resto de fora.

### `get_observation_points(seg, brain, cfg, rng)`
Produz os pontos "obs" (supervisão por intervalo):
1. Mapeia os voxels do cérebro para as três classes.
2. Amostra todos (ou `max_tumor_obs`) voxels tumorais.
3. Define uma faixa "próxima do tumor" (distância ≤ `near_tumor_mm`) com transformada de distância euclidiana.
4. Amostra voxels saudáveis na mesma quantidade dos tumorais, com `near_fraction` deles dentro da faixa próxima.
5. Para cada voxel, grava o intervalo `(lower, upper)` conforme sua classe.

### `get_collocation_points(brain, seg, cfg, rng)`
Produz os pontos onde a EDP é cobrada (sem valor-alvo de `u`):
1. Calcula a faixa próxima do tumor (igual à de observação).
2. Amostra `n_collocation` voxels do cérebro, com viés para a faixa próxima.
3. Desloca aleatoriamente cada ponto em `[-0.5, 0.5]` dentro de seu voxel (a física vale no contínuo).

### `get_boundary_points(brain, affine, cfg, rng)`
Produz os pontos da superfície do cérebro para a condição de contorno Neumann (sem fluxo):
1. Superfície = voxels do cérebro que somem após erosão binária de 1 voxel.
2. Amostra `n_boundary` voxels dessa superfície.
3. Suaviza a máscara binária com Gaussiana (σ = 2) para ter um gradiente numérico estável.
4. Normal externa = `-gradiente` (máscara decresce para fora), rotacionada para mm pela affine e normalizada.

---

## 3. `main.py` — Geração dos datasets reais (pacientes)

Script que percorre `data/Patient-*`, extrai os pontos para a PINN e grava `processed/pinn/<paciente>.npz` + `.json`.

Passo a passo da `main()`:

1. **Configuração**: instancia `PINNDataConfig` e cria o diretório de saída.
2. **Loop por paciente**:
    1. Cria um RNG determinístico com a seed da config.
    2. `get_week_indexer(patient.weeks)` → mapa `{nome_semana: t ∈ [0, 1]}` e a duração total (em semanas). Para uma única semana, `t = 0`.
    3. **Observações (loop por semana)**: para cada semana:
        - Carrega segmentação, máscara cerebral, canais de features (t1, t2, ct1, flair) e affine via `load_week_data`.
        - Na primeira semana, pinta o `CoordinateFrame` compartilhado e guarda a máscara cerebral.
        - Amostra pontos de observação via `pinn.get_observation_points` (voxels, labels, bounds).
        - Converte voxels para coordenadas normalizadas, adiciona a coluna de tempo `t` daquela semana.
        - Guarda arrays `xyzt`, `lower`, `upper`, `labels`, `features`.
    4. Concatena os arrays de todas as semanas.
    5. **Pontos de colocação**: usa a união das segmentações como "região do tumor" para focar amostragem. Perturba os voxels com ruído uniforme e amostra `t` uniformemente em `[0, 1]`.
    6. **Pontos de borda**: amostra na superfície da máscara cerebral da primeira semana com normais externas; `t` também sorteado uniformemente.
    7. **Empacotamento**: achata `obs`, `col`, `bnd` em um único dicionário `arrays` com chaves prefixadas (`obs_xyzt`, `col_xyzt`, ...).
    8. **Seed guess**: centróide do core tumoral (realce + necrose) na primeira semana, em coordenadas normalizadas — serve de palpite inicial para a origem do tumor.
    9. **Metadata**: dicionário com nome do paciente, lista de semanas, duração, affine, centro, escala, e contagens de pontos.
    10. **Gravação**: `np.savez_compressed` para arrays + `json.dump` para o metadata.

### Funções auxiliares

- `get_week_indexer(weeks)`: parseia nomes como `week-2-1` em `(semana, sub-semana)`, calcula posições temporais e normaliza para `[0, 1]`. Retorna o mapa e a duração absoluta.
- `parse_week(name)`: regex `^week-(\d+)(?:-(\d+))?$` → `(semana, sub-semana)`.
- `load_week_data(week)`: carrega os 4 canais de MRI (t1, t2, ct1, flair) normalizados pelo cérebro, a máscara `brain`, a segmentação e a affine.
- `load_nib_file(path)`: carrega NIfTI em orientação canônica (RAS).
- `normalize(raw, brain_mask)`: clip por percentil (0.5, 99.5), z-score dentro do cérebro, zera fora.

---

## 4. `simulator.py` — Dataset sintético (teste)

Simula numericamente a equação Fisher-Kolmogorov 3D num cubo com condição Neumann e empacota o resultado no mesmo esquema de `main.py`, mas com `obs_target` (valor exato de `u`) além dos intervalos. Isso permite medir o erro de recuperação dos parâmetros contra um ground-truth conhecido.

### Equação discretizada (Crank-Nicolson semi-implícito)
```
(I - 0.5*dt*D*L) u^{n+1} = (I + 0.5*dt*D*L) u^n + dt * rho * u^n * (1 - u^n)
```
Difusão implícita (estável), reação explícita. Neumann implementada via ghost-points.

### Passo a passo da `main()`
1. Cria `SimConfig` com os ground-truths (D = 0.10 mm²/dia, rho = 0.05 /dia), domínio cúbico de ±90 mm e 70 dias.
2. `simulate(cfg)`:
    - Monta o laplaciano 3D com condição Neumann.
    - Inicializa `u` como uma gaussiana isotrópica (σ = 10 mm) no centro.
    - Fatora a matriz implícita uma vez com `splu`.
    - Loop temporal: calcula o RHS e resolve o sistema; clipa `u` em `[0, 1]`.
3. `to_pinn_arrays(history, axis_mm, cfg)`:
    - **Obs**: sorteia `n_obs` quádruplas `(t, i, j, k)`, lê `u_true` no histórico, converte coordenadas para normalizadas, atribui rótulos e bandas `(lower, upper)` conforme o valor de `u_true`.
    - **Col**: sorteio uniforme em `[-1, 1]³ × [0, 1]`.
    - **Bnd**: `sample_cube_boundary` sorteia pontos nas 6 faces do cubo com as respectivas normais externas.
4. `metadata(cfg)`: inclui `diffusion_norm` e `proliferation_norm` (parâmetros convertidos para a escala normalizada que a PINN enxerga) como ground-truth.
5. Grava `data_sim/test_synthetic.npz` + `.json`.

### Normalizações
- Espacial: `x_norm = x_mm / scale_mm` (coincide com `main.py`).
- Temporal: `t_norm = t_dia / time_span_days`.
- Parâmetros: `D_norm = D_phys * time_span / scale² ` e `rho_norm = rho_phys * time_span`.

### Funções auxiliares
- `build_laplacian(n, dx)`: produto de Kronecker do laplaciano 1D consigo mesmo três vezes; as bordas do 1D viram `[-2, +2]` (ghost-point Neumann).
- `simulate(cfg)`: integração temporal.
- `sample_cube_boundary(n, rng)`: amostragem uniforme nas 6 faces do cubo.
- `to_pinn_arrays`: empacotamento no esquema da PINN.
- `metadata(cfg)`: dicionário com ground-truth e config.

---

## 5. `train.py` — Treinamento da PINN

Carrega um dataset (real ou sintético), treina um MLP com três losses e grava um `.pt` com pesos e histórico. Os parâmetros `D` e `rho` são aprendidos em paralelo com os pesos (problema inverso), parametrizados em log para garantir positividade.

### `TrainConfig` (dataclass)
- Arquitetura: `hidden_width=128`, `hidden_depth=3`, `tanh`.
- Learning rates separadas: `model_lr=1e-3` para a MLP, `param_lr=1e-2` para `log_D` e `log_rho`.
- 10 000 épocas, log a cada 500.
- Pesos das losses: `data_weight = residual_weight = boundary_weight = 1.0`.
- Palpites iniciais: `d_init=1e-3`, `rho_init=1.0` (ordem de magnitude típica de glioma).
- Device: autodetecta CUDA > MPS > CPU.

### `PINN(nn.Module)`
- MLP: `Linear(4, W) → Tanh → [Linear(W, W) → Tanh] × (depth-1) → Linear(W, 1)`.
- Parâmetros físicos: `log_d` e `log_rho` aprendíveis; `D = exp(log_d)`, `rho = exp(log_rho)`.
- `forward(xyzt)`: retorna `u` escalar.

### Losses

1. **`data_loss`**:
    - Se há `obs_target` (dataset sintético): `MSE(u_pred, u_target)`.
    - Caso contrário (dataset real): `mean(relu(lower - u)² + relu(u - upper)²)` — penaliza só quando sai da banda.

2. **`residual_loss`** (Fisher-Kolmogorov):
    - Nos pontos de colocação, computa `u_t`, `u_xx`, `u_yy`, `u_zz` via autograd.
    - Resíduo: `u_t - D * (u_xx + u_yy + u_zz) - rho * u * (1 - u)`.
    - Loss = média do resíduo ao quadrado.

3. **`boundary_loss`** (Neumann):
    - Nos pontos de borda, `(grad(u)[:3] · normal)²` — derivada temporal `d/dt` é descartada pois não entra na condição de contorno.

### `train(cfg, data, meta)` — passo a passo
1. Fixa a seed, instancia a PINN e envia para o device.
2. Otimizador Adam com dois grupos de parâmetros (MLP e `(log_d, log_rho)`), lrs separados.
3. Inicializa o dicionário `history` para `loss`, `data`, `residual`, `boundary`, `D_norm`, `rho_norm`.
4. **Loop de treino** (`cfg.n_epochs` épocas):
    1. `zero_grad`.
    2. Calcula as três losses.
    3. Soma ponderada → `loss`.
    4. `loss.backward()` + `optimizer.step()`.
    5. Armazena métricas no histórico.
    6. A cada `log_every` épocas (ou na última), imprime `ep X loss=... data=... res=... bnd=... D=... rho=...`.
5. Chama `report_recovery(model, meta)`: imprime `D_norm`, `rho_norm` aprendidos, compara com ground-truth (se existir) e converte para unidades físicas.

### `train_one(npz_path, cfg)`
1. Carrega dataset via `load_dataset` (cria tensores no device; flagfa `requires_grad=True` nos pontos de colocação e borda).
2. Imprime informações do dataset (contagem de pontos, modo da data loss).
3. Chama `train(cfg, data, meta)`.
4. Salva `state_dict`, `history`, e `config` em `<dataset>_trained.pt`.

### `main()` (CLI)
1. Argparse: `dataset` (arquivo ou diretório), `--epochs`, `--seed`, `--d-init`, `--rho-init`, `--device`.
2. `collect_datasets(path)`: lista todos os `.npz` com `.json` correspondente, pulando os `*_trained`.
3. Para cada dataset, chama `train_one`.

### `report_recovery(model, meta)`
- Imprime `D_norm` e `rho_norm` aprendidos.
- Se houver `ground_truth` na metadata, imprime os valores verdadeiros e o erro absoluto.
- Se houver `scale_mm` e `time_span_*` na metadata, converte para `D_phys` (mm²/dia ou mm²/semana) e `rho_phys` (/dia ou /semana).

---

## 6. `plots.py` — Visualização dos resultados

Lê CSVs de treinamento salvos em `results/` e gera gráficos de `rho`, `D` e `loss` por época, comparados com os valores reais (quando aplicável).

Passo a passo:

1. Define os ground-truths de referência (`REAL_RHO = 3.5`, `REAL_D = 0.000864198`) e a lista de rodadas `RESULTS` (`original_20k`, `original_30k`, `enhanced_10k`, `enhanced_30k`).
2. `main()`:
    - Para cada rodada sintética, lê `results/<rodada>.csv` (separador `;`) e chama `plot(...)` três vezes: `rho` (com linha real), `D` (com linha real), `loss` (sem referência real).
    - Também plota o resultado da rodada real (`real_10k.csv`) para `rho`, `D` e `loss`, sem linha de referência.
3. `plot(result, result_file, value_type, real, scale)`:
    - `x` = coluna `epoch`, `y` = coluna do parâmetro.
    - Se `real` não é `None`, plota a linha horizontal pontilhada do valor real.
    - Plota a curva estimada, ajusta `ylim`, legenda e salva em `figures/<rodada>-<parâmetro>.png`.

---

## Resumo do pipeline end-to-end

1. **Preparar dados reais**: `python main.py` → gera `processed/pinn/Patient-*.npz` e `.json`.
2. **Preparar dado sintético (teste)**: `python simulator.py` → gera `data_sim/test_synthetic.npz` e `.json`.
3. **Treinar**: `python train.py <arquivo-ou-pasta> [--epochs N]` → gera `*_trained.pt` ao lado de cada dataset.
4. **Visualizar**: `python plots.py` → gera PNGs em `figures/` a partir dos CSVs salvos em `results/`.
