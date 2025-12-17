import matplotlib.pyplot as plt
import os

import pandas as pd
def Visulize_ordinal(df, m):

    value_mappings = {}
    for col_idx, col in enumerate(df.columns):
        unique_values = sorted(df[col].unique())
        new_values = []
        
        for i in range(len(unique_values)):
            if i < len(unique_values) - 1:
                new_value = 0.5 * (m[col_idx][i] + m[col_idx][i + 1])
            else:
                new_value = m[col_idx][i]
            new_values.append(new_value)

        value_mappings[col] = {unique_values[i]: new_values[i] for i in range(len(unique_values))}

    return value_mappings

def plot_value_mappings(value_mappings, save_dir='plots'):


    os.makedirs(save_dir, exist_ok=True)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  

    for idx, (col, mapping) in enumerate(value_mappings.items()):
        original_values = list(mapping.keys())
        updated_values = list(mapping.values())

        plt.figure(figsize=(10, 6))
        plt.plot(original_values, updated_values, marker='o', color=colors[idx % len(colors)])  # Use modulo to cycle colors

        plt.title(f'{col}')
        plt.grid(True)
        

        plt.xticks(ticks=original_values, labels=[str(v) for v in original_values], fontsize=14)

        filename = f'{save_dir}/{col}_value_mapping.png'
        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
# Example usage:
    X_train = pd.DataFrame([[1, 0.2, 3], [1.5, 0.2, 3], [5, 8, 9]], columns=['col1', 'col2', 'col3'])
    m = [[1, 2, 3, 4], [0.2, 0.3, 8, 10], [3, 3, 9, 11]]

    value_mappings = Visulize_ordinal(X_train, m)
    plot_value_mappings(value_mappings)

