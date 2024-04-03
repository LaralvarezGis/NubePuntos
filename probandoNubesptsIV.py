import pandas as pd
import plotly.express as px

# Importar los datos
data = pd.read_csv('C:/Users/alvar/OneDrive/Escritorio/Lara tareas/Datos_GPRComaModificado.csv')

# Preparar los datos
x = data['x']
y = data['y']
z_anomaly = -data['z_anomaly']  # Multiplicar por -1 para invertir los valores

# Crear la nube de puntos
fig = px.scatter_3d(data, x=x, y=y, z=z_anomaly, color='z_anomaly', title='Nube de puntos 3D')

# Mostrar el gráfico
fig.show()