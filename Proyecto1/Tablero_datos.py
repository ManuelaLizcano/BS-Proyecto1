import dash
from dash import dcc  # dash core components 
from dash import html # dash html components
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

# Cargar datos
file_path = r"C:\Users\USUARIO\Documents\BS-Proyecto1\Proyecto1\x_test.csv"
df = pd.read_csv(file_path)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Diseño del tablero
app.layout = html.Div([
    html.H1("Tablero de Datos - Alquiler de Apartamentos"),
    
    html.Div([
        html.Label("Ciudad:"),
        dcc.Dropdown(
            id='cityname',
            options=[{'label': loc, 'value': loc} for loc in df['cityname'].unique()],
            placeholder="Seleccione una ciudad"
        ),
        html.Label("Número de habitaciones:"),
        dcc.Input(id='habitaciones', type='number', placeholder='Número de habitaciones'),
        html.Label("Área en m²:"),
        dcc.Input(id='area', type='number', placeholder='Área en m²'),
        html.Label("Precio máximo:"),
        dcc.Input(id='precio_max', type='number', placeholder='Precio máximo'),
        html.Button('Filtrar', id='filtrar-btn', n_clicks=0),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    html.Div([
        dcc.Graph(id='histograma_precios'),
        dcc.Graph(id='scatter_precio_area')
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'}),
    
    html.Div([
        html.H3("Predicción de Precio"),
        dcc.Input(id='input_prediccion', type='number', placeholder='Ingrese características'),
        html.Button('Predecir', id='predecir-btn', n_clicks=0),
        html.Div(id='resultado_prediccion')
    ], style={'marginTop': '30px'})
])

# Callback para actualizar gráficos
@app.callback(
    [Output('histograma_precios', 'figure'),
     Output('scatter_precio_area', 'figure')],
    [Input('filtrar-btn', 'n_clicks')],
    [dash.State('cityname', 'value'),
     dash.State('habitaciones', 'value'),
     dash.State('area', 'value'),
     dash.State('precio_max', 'value')]
)
def actualizar_graficos(n_clicks, cityname, habitaciones, area, precio_max):
    df_filtrado = df.copy()
    
    if cityname:
        df_filtrado = df_filtrado[df_filtrado['cityname'] == cityname]
    if habitaciones:
        df_filtrado = df_filtrado[df_filtrado['bedrooms'] == habitaciones]
    if area:
        df_filtrado = df_filtrado[df_filtrado['square_feet'] <= area]
    if precio_max:
        df_filtrado = df_filtrado[df_filtrado['price'] <= precio_max]
    
    # Manejo de DataFrame vacío
    if df_filtrado.empty:
        hist_fig = px.histogram(x=[], title="No hay datos disponibles")
        scatter_fig = px.scatter(x=[], y=[], title="No hay datos disponibles")
    else:
        hist_fig = px.histogram(df_filtrado, x='price', title='Distribución de Precios')
        scatter_fig = px.scatter(df_filtrado, x='square_feet', y='price', title='Relación entre Área y Precio')
    
    return hist_fig, scatter_fig

if __name__ == '__main__':
    app.run_server(debug=True)
