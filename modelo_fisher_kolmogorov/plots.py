import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

REAL_RHO = 3.5
REAL_D = 0.000864198

PATH = Path('results')
RESULTS = ['original_20k', 'original_30k', 'enhanced_10k', 'enhanced_30k']

def main(): 
    for result_file in RESULTS:
        result = pd.read_csv(PATH / f'{result_file}.csv', sep=';', dtype=np.float32)
        
        plot(result, result_file, 'rho', REAL_RHO, (0,4))
        plot(result, result_file, 'D', REAL_D, (0, 0.001))
        plot(result, result_file, 'loss', None, (0, 0.001))

    real = pd.read_csv(PATH / f'real_10k.csv', sep=';', dtype=np.float32)

    plot(real, 'real_10k', 'rho', None, (0,1))
    plot(real, 'real_10k', 'D', None, (0, 0.001))
    plot(real, 'real_10k', 'loss', None, (0, 0.15))

def plot(result, result_file, value_type, real, scale):
    x = result['epoch']
    y = result[value_type]

    fig, ax = plt.subplots()

    if real is not None:
        ax.plot(x, [real for _ in x], '--', label=f'{value_type} real')
    ax.plot(x, y, label=f'{value_type} estimado')

    ax.set(
        ylim=scale, 
        xlabel='epochs', ylabel=value_type
    )
    
    ax.legend()

    fig.savefig(f'figures/{result_file}-{value_type}.png')

if __name__ == '__main__':
    main()