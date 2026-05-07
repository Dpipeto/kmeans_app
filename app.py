from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'clientes.csv')

def to_py(v):
    if hasattr(v, 'item'): return v.item()
    return v

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    features = ['edad','ingreso_mensual','gasto_mensual','frecuencia_compras','antiguedad_meses','num_productos']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled, scaler, features

@app.route('/')
def index(): return render_template('index.html')
@app.route('/dataset')
def dataset(): return render_template('dataset.html')
@app.route('/conceptos')
def conceptos(): return render_template('conceptos.html')
@app.route('/modelo')
def modelo(): return render_template('modelo.html')
@app.route('/resultados')
def resultados(): return render_template('resultados.html')

@app.route('/api/dataset-info')
def api_dataset_info():
    df, X, X_scaled, scaler, features = load_and_prepare_data()
    stats = {}
    for col in features:
        stats[col] = {
            'mean': round(float(df[col].mean()), 2),
            'std':  round(float(df[col].std()), 2),
            'min':  round(float(df[col].min()), 2),
            'max':  round(float(df[col].max()), 2),
        }
    preview = [{k: to_py(v) for k, v in row.items()} for row in df.head(10).to_dict(orient='records')]
    return jsonify({'stats': stats, 'preview': preview, 'total': len(df), 'features': features})

@app.route('/api/elbow')
def api_elbow():
    df, X, X_scaled, scaler, features = load_and_prepare_data()
    inertias, k_range = [], list(range(1, 11))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(round(float(km.inertia_), 4))
    return jsonify({'k_values': k_range, 'inertias': inertias})

@app.route('/api/cluster')
def api_cluster():
    k = max(2, min(int(request.args.get('k', 3)), 8))
    df, X, X_scaled, scaler, features = load_and_prepare_data()

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    centroids_2d = pca.transform(km.cluster_centers_)
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)

    centroid_rows = []
    for i, c in enumerate(centroids_orig):
        row = {'cluster': i}
        for j, f in enumerate(features):
            row[f] = round(float(c[j]), 2)
        centroid_rows.append(row)

    points = []
    for i in range(len(df)):
        r = df.iloc[i]
        points.append({
            'x': round(float(X_2d[i,0]), 4), 'y': round(float(X_2d[i,1]), 4),
            'cluster': int(labels[i]),
            'cliente_id': str(r['cliente_id']),
            'edad': int(r['edad']),
            'ingreso': int(r['ingreso_mensual']),
            'gasto': int(r['gasto_mensual']),
        })

    df_out = df.copy()
    df_out['cluster'] = labels
    table = [{k: to_py(v) for k, v in row.items()} for row in df_out.head(50).to_dict(orient='records')]
    sizes = {str(int(lbl)): int((labels == lbl).sum()) for lbl in range(k)}

    return jsonify({
        'k': k, 'points': points,
        'centroids_2d': [{'x': round(float(c[0]),4), 'y': round(float(c[1]),4), 'cluster': i} for i,c in enumerate(centroids_2d)],
        'centroids_orig': centroid_rows, 'table': table,
        'sizes': sizes, 'features': features,
        'inertia': round(float(km.inertia_), 4),
        'var_explained': round(float(sum(pca.explained_variance_ratio_))*100, 1),
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
