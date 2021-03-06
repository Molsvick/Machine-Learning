import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%matplotlib inline


#Generacion de datos

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Crear datos aleatorios y almacenar en la matriz X y el vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation,
                      cluster_std=clusterDeviation)

    # Estandarizar características eliminando el promedio y la unidad de escala
    X = StandardScaler().fit_transform(X)
    return X, y

X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

#Modelado
epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_


# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        # El color negro se utilizará para señalar ruido en los datos.
        col = 'k'

    class_member_mask = (labels == k)

    # Dibujar los datos que estan agrupados (clusterizados)
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)

    # Dibujar los valores atípicos
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)