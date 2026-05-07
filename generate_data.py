import numpy as np
import pandas as pd

np.random.seed(42)

n = 150

# Group 1: Young low-income customers
g1 = pd.DataFrame({
    'edad': np.random.normal(24, 4, 40).astype(int),
    'ingreso_mensual': np.random.normal(1200000, 150000, 40),
    'gasto_mensual': np.random.normal(400000, 80000, 40),
    'frecuencia_compras': np.random.normal(3, 1, 40),
    'antiguedad_meses': np.random.normal(8, 3, 40),
    'num_productos': np.random.normal(1.5, 0.5, 40),
})

# Group 2: Middle-aged medium-income customers
g2 = pd.DataFrame({
    'edad': np.random.normal(38, 6, 55).astype(int),
    'ingreso_mensual': np.random.normal(3500000, 300000, 55),
    'gasto_mensual': np.random.normal(1200000, 200000, 55),
    'frecuencia_compras': np.random.normal(7, 2, 55),
    'antiguedad_meses': np.random.normal(24, 8, 55),
    'num_productos': np.random.normal(3.2, 0.7, 55),
})

# Group 3: Older high-income customers
g3 = pd.DataFrame({
    'edad': np.random.normal(52, 7, 55).astype(int),
    'ingreso_mensual': np.random.normal(7000000, 800000, 55),
    'gasto_mensual': np.random.normal(2800000, 400000, 55),
    'frecuencia_compras': np.random.normal(12, 3, 55),
    'antiguedad_meses': np.random.normal(48, 12, 55),
    'num_productos': np.random.normal(5.5, 1.0, 55),
})

df = pd.concat([g1, g2, g3], ignore_index=True)

# Clean up negatives
df['edad'] = df['edad'].clip(18, 75)
df['ingreso_mensual'] = df['ingreso_mensual'].clip(700000, 15000000).round(-3)
df['gasto_mensual'] = df['gasto_mensual'].clip(100000, 5000000).round(-3)
df['frecuencia_compras'] = df['frecuencia_compras'].clip(1, 20).round().astype(int)
df['antiguedad_meses'] = df['antiguedad_meses'].clip(1, 84).round().astype(int)
df['num_productos'] = df['num_productos'].clip(1, 8).round().astype(int)

# Add customer ID
df.insert(0, 'cliente_id', [f'CLI-{str(i+1).zfill(3)}' for i in range(len(df))])

df.to_csv('/home/claude/kmeans_app/data/clientes.csv', index=False)
print(f"Dataset generado: {len(df)} registros")
print(df.describe())
