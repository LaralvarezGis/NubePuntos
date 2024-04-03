import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from scipy.interpolate import griddata
import plotly.graph_objects as go


df = pd.read_csv('C:/Users/alvar/OneDrive/Escritorio/Lara tareas/Datos_GPR.csv')


# Calcular la profundidad del suelo como la diferencia entre "z_anomaly" y "z"
df['profundidad_suelo'] = df['z'] - df['z_anomaly']

# Imprimir los primeros registros del DataFrame para verificar
print(df.head())

# Visualizar los datos (opcional)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], c=df['profundidad_suelo'], cmap='viridis', marker='o', edgecolors='k')
plt.colorbar(label='Profundidad del suelo')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Estimación de profundidad del suelo')
plt.grid(True)
plt.show()

# Crear una figura y un eje 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos de profundidad como una superficie tridimensional
x = df['x']
y = df['y']
z = np.zeros_like(x)  # El plano z=0 será la superficie del suelo
profundidad = df['profundidad_suelo']

# Crear la superficie tridimensional
ax.plot_trisurf(x, y, profundidad, cmap='viridis', edgecolor='none')

# Etiquetas y título
ax.set_xlabel('Coordenada X')
ax.set_ylabel('Coordenada Y')
ax.set_zlabel('Profundidad del suelo')
ax.set_title('Mapa 3D de profundidad del suelo')

# Mostrar la visualización
plt.show()





from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Supongamos que tienes tus datos de profundidad estimada en un DataFrame df
# con columnas 'x', 'y' (coordenadas) y 'profundidad_suelo' (profundidad estimada)
# Puedes ajustar esto según la estructura real de tus datos

# Generar una cuadrícula de puntos para la interpolación
x_min, x_max = df['x'].min(), df['x'].max()
y_min, y_max = df['y'].min(), df['y'].max()
x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Interpolar la profundidad del suelo en la cuadrícula utilizando Kriging
points = df[['x', 'y']].values
values = df['profundidad_suelo'].values
depth_grid = griddata(points, values, (x_mesh, y_mesh), method='linear')

# Crear figura tridimensional con Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie interpolada
surf = ax.plot_surface(x_mesh, y_mesh, depth_grid, cmap='viridis')

# Configurar etiquetas y título
ax.set_xlabel('Coordenada X')
ax.set_ylabel('Coordenada Y')
ax.set_zlabel('Profundidad del suelo')
ax.set_title('Mapa 3D interpolado de profundidad del suelo')

# Mostrar la barra de color
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mostrar gráfico
plt.show()
