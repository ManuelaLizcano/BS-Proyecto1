import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# Datos para visualización (formulario y gráficos)
# =============================================================================
df = pd.read_csv("BS-Proyecto1-main/Proyecto1/x_test.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.rename(columns={'real_price': 'price'}, inplace=True)

# =============================================================================
# Datos y modelo para la predicción (regresión con 10 variables)
# Se utiliza un segundo CSV que contiene las 10 variables, además de "id", "cityname", "state" y "price"
# =============================================================================
df1 = pd.read_csv("BS-Proyecto1-main/Proyecto1/df1.csv")
df1.columns = df1.columns.str.strip().str.lower().str.replace(" ", "_")

# Extraer columnas que no se usarán en la regresión
id_column = df1['id']
df1 = df1.drop('id', axis=1)
cityname_column = df1['cityname']
df1 = df1.drop('cityname', axis=1)
state_column = df1['state']
df1 = df1.drop('state', axis=1)

# Variables predictoras y variable respuesta
X = df1.drop("price", axis=1)  # Se asume que X tiene 10 columnas numéricas
y = df1["price"].values.reshape(-1, 1)

# Escalado por separado
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Volver a agregar las columnas no escaladas para identificación (no se usan para entrenar)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df['id'] = id_column.values
X_scaled_df['cityname'] = cityname_column.values
X_scaled_df['state'] = state_column.values

# Lista de las 10 variables (las que usaremos para la predicción)
features = list(X.columns)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_scaled, test_size=0.3, random_state=42)

# Entrenar modelo RandomForest (se usa sobre las 10 variables)
modelo_rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
modelo_rf.fit(X_train.drop(["id", "cityname", "state"], axis=1), y_train.ravel())

# =============================================================================
# Layout del Dash
# =============================================================================
app = dash.Dash(__name__)  # Aquí se usa __name_, con dos guiones bajos

app.layout = html.Div(
    style={'backgroundColor': '#E3F2FD', 'fontFamily': 'Arial', 'padding': '20px'},
    children=[
        html.H1("BS - Alquiler de Apartamentos", style={'textAlign': 'center', 'color': '#0D47A1'}),
        # Sección de predicción
        html.Div(
            style={
                'width': '30%',
                'backgroundColor': '#BBDEFB',
                'padding': '20px',
                'borderRadius': '5px',
                'marginBottom': '20px'
            },
            children=[
                html.H3("Predicción del Precio con Random Forest (10 variables)",
                        style={'color': '#1565C0'}),
                html.P("Ingrese los valores de las siguientes variables:",
                       style={'color': '#0D47A1'}),
                # Generar dinámicamente un input para cada variable
                html.Div(
                    [
                        html.Div([
                            html.Label(f"{feature.capitalize()}:", style={'color': '#0D47A1'}),
                            dcc.Input(
                                id=f'predict_{feature}',
                                type='number',
                                placeholder=f'Ingrese {feature}',
                                style={'backgroundColor': 'white', 'marginBottom': '10px'}
                            )
                        ])
                        for feature in features
                    ],
                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}
                ),
                html.Button('Predecir Precio',
                            id='btn_predecir',
                            n_clicks=0,
                            style={
                                'marginTop': '10px',
                                'backgroundColor': '#1976D2',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px'
                            }),
                html.Div(
                    id='resultado_prediccion',
                    style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': 'bold', 'color': '#0D47A1'}
                )
            ]
        ),
        # Resto del layout (gráficos)
        html.Div(
            style={'width': '65%', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
            children=[
                html.Div(
                    style={'width': '48%', 'backgroundColor': '#BBDEFB', 'padding': '20px',
                           'borderRadius': '5px', 'marginBottom': '20px'},
                    children=[
                        html.H3("Filtros para Histograma", style={'color': '#1565C0'}),
                        html.P("Este gráfico muestra la distribución de los precios de alquiler en función de la ciudad y el precio máximo seleccionado.",
                               style={'color': '#0D47A1'}),
                        html.Label("Ciudad:", style={'color': '#0D47A1'}),
                        dcc.Dropdown(
                            id='cityname_hist',
                            options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                            placeholder="Seleccione una ciudad",
                            style={'backgroundColor': 'white'}
                        ),
                        html.Label("Precio máximo:", style={'color': '#0D47A1'}),
                        dcc.Dropdown(
                            id='precio_max_hist',
                            options=[{'label': f"${p:,.0f}", 'value': p} for p in sorted(df['price'].dropna().unique())],
                            placeholder="Seleccione el precio máximo",
                            style={'backgroundColor': 'white'}
                        ),
                        dcc.Graph(id='histograma_precios')
                    ]
                ),
                html.Div(
                    style={'width': '48%', 'backgroundColor': '#BBDEFB', 'padding': '20px',
                           'borderRadius': '5px', 'marginBottom': '20px'},
                    children=[
                        html.H3("Filtros para Scatter Plot", style={'color': '#1565C0'}),
                        html.P("Este gráfico muestra la relación entre el área en m² y el precio, permitiendo filtrar por número de habitaciones y área seleccionada.",
                               style={'color': '#0D47A1'}),
                        html.Label("Número de habitaciones:", style={'color': '#0D47A1'}),
                        dcc.Dropdown(
                            id='habitaciones_scatter',
                            options=[{'label': str(h), 'value': h} for h in sorted(df['bedrooms'].dropna().unique())],
                            placeholder="Seleccione el número de habitaciones",
                            style={'backgroundColor': 'white'}
                        ),
                        dcc.Graph(id='scatter_precio_area')
                    ]
                )
            ]
        ),
        html.Div(
            style={'width': '100%', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
            children=[
                html.Div(
                    style={'width': '48%', 'backgroundColor': '#BBDEFB', 'padding': '20px',
                           'borderRadius': '5px', 'marginBottom': '20px'},
                    children=[
                        html.H3("Filtros para Heatmap de Ubicación", style={'color': '#1565C0'}),
                        html.P("Este heatmap muestra la distribución geográfica de los precios de alquiler, basándose en la ciudad seleccionada.",
                               style={'color': '#0D47A1'}),
                        html.Label("Ciudad:", style={'color': '#0D47A1'}),
                        dcc.Dropdown(
                            id='cityname_heatmap',
                            options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                            placeholder="Seleccione una ciudad",
                            style={'backgroundColor': 'white'}
                        ),
                        dcc.Graph(id='heatmap_precio_ubicacion')
                    ]
                ),
                html.Div(
                    style={'width': '48%', 'backgroundColor': '#BBDEFB', 'padding': '20px',
                           'borderRadius': '5px', 'marginBottom': '20px'},
                    children=[
                        html.H3("Heatmap de Precio vs Variable Elegida", style={'color': '#1565C0'}),
                        html.P("Este heatmap permite visualizar cómo varía el precio en función de la variable seleccionada, como área, número de habitaciones, latitud o longitud.",
                               style={'color': '#0D47A1'}),
                        html.Label("Seleccione una variable:", style={'color': '#0D47A1'}),
                        dcc.Dropdown(
                            id='heatmap_variable',
                            options=[
                                {'label': 'Área (m²)', 'value': 'square_feet'},
                                {'label': 'Número de habitaciones', 'value': 'bedrooms'},
                                {'label': 'Latitud', 'value': 'latitude'},
                                {'label': 'Longitud', 'value': 'longitude'}
                            ],
                            placeholder="Seleccione una variable",
                            style={'backgroundColor': 'white'}
                        ),
                        dcc.Graph(id='heatmap_precio_variable')
                    ]
                )
            ]
        )
    ]
)

# =============================================================================
# Callback de predicción utilizando las 10 variables
# =============================================================================
@app.callback(
    Output('resultado_prediccion', 'children'),
    [Input('btn_predecir', 'n_clicks')],
    [State(f'predict_{feature}', 'value') for feature in features]
)
def predecir_precio(n_clicks, *feature_values):
    if n_clicks > 0 and all(v is not None for v in feature_values):
        # Crear DataFrame de entrada con 1 registro
        entrada = pd.DataFrame([list(feature_values)], columns=features)
        # Escalar la entrada usando el mismo escalador
        entrada_scaled = scaler_X.transform(entrada)
        pred_scaled = modelo_rf.predict(entrada_scaled)
        # Desescalar la predicción
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return f"Precio estimado: ${pred:,.2f}"
    return "Ingrese los valores y presione Predecir Precio"

# =============================================================================
# Callbacks para actualizar los gráficos (se utilizan los datos originales)
# =============================================================================
@app.callback(
    Output('histograma_precios', 'figure'),
    [Input('cityname_hist', 'value'),
     Input('precio_max_hist', 'value')]
)
def actualizar_histograma(cityname, precio_max):
    df_filtrado = df.copy()
    if cityname:
        df_filtrado = df_filtrado[df_filtrado['cityname'] == cityname]
    if precio_max is not None:
        df_filtrado = df_filtrado[df_filtrado['price'] <= precio_max]
    if df_filtrado.empty:
        return px.histogram(x=[], title='No hay datos disponibles')
    return px.histogram(df_filtrado, x='price', title='Distribución de Precios')

@app.callback(
    Output('scatter_precio_area', 'figure'),
    [Input('habitaciones_scatter', 'value')]
)
def actualizar_scatter(habitaciones):
    df_filtrado = df.copy()
    if df_filtrado.empty:
        return px.scatter(x=[0], y=[0], title='No hay datos disponibles')
    if habitaciones is not None:
        df_filtrado = df_filtrado[df_filtrado['bedrooms'] == habitaciones]
    if df_filtrado.empty:
        return px.scatter(x=[0], y=[0], title='No hay datos disponibles')
    return px.scatter(df_filtrado, x='square_feet', y='price', title='Relación entre Área y Precio')

@app.callback(
    Output('heatmap_precio_ubicacion', 'figure'),
    [Input('cityname_heatmap', 'value')]
)
def actualizar_heatmap_precio_ubicacion(cityname):
    df_filtrado = df.copy()
    if cityname:
        df_filtrado = df_filtrado[df_filtrado['cityname'] == cityname]
    if df_filtrado.empty:
        return px.density_heatmap(x=[], y=[], title='No hay datos disponibles')
    return px.density_heatmap(df_filtrado, x='longitude', y='latitude', z='price',
                              title='Mapa de Calor - Precio por Ubicación',
                              labels={'longitude': 'Longitud', 'latitude': 'Latitud', 'price': 'Precio'},
                              color_continuous_scale='OrRd')

@app.callback(
    Output('heatmap_precio_variable', 'figure'),
    [Input('heatmap_variable', 'value')]
)
def actualizar_heatmap_variable(variable):
    if not variable:
        return px.density_heatmap(x=[0], y=[0], z=[0], title='No hay datos disponibles')
    
    df_filtrado = df.dropna(subset=[variable, 'price'])
    if df_filtrado.empty:
        return px.density_heatmap(x=[0], y=[0], z=[0], title='No hay datos disponibles')
    return px.density_heatmap(df_filtrado, x=variable, y='price', z='price',
                              title=f'Mapa de Calor - Precio vs {variable.capitalize()}',
                              labels={variable: variable.capitalize(), 'price': 'Precio'},
                              color_continuous_scale='OrRd')

# =============================================================================
# Ejecutar la aplicación
# =============================================================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)