import dash
from dash import dcc  # dash core components
from dash import html  # dash html components
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Cargar datos
file_path="BS-Proyecto1-main/Proyecto1/x_test.csv"




df = pd.read_csv(file_path)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Asegurar que la columna de precios existe y es consistente
df.rename(columns={'real_price': 'price'}, inplace=True)

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

app.layout = html.Div(
    style={'backgroundColor': '#E3F2FD', 'fontFamily': 'Arial', 'padding': '20px'},
    children=[
        html.H1("BS - Alquiler de Apartamentos",
                style={'textAlign': 'center', 'color': '#0D47A1'}),
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
            children=[
                # Columna de Predicción
                html.Div(
                    style={
                        'width': '30%',
                        'backgroundColor': '#BBDEFB',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'marginBottom': '20px'
                    },
                    children=[
                        html.H3("Predicción del Precio con Random Forest",
                                style={'color': '#1565C0'}),
                        html.Div([
                            html.Label("Número de habitaciones:",
                                       style={'color': '#0D47A1'}),
                            dcc.Dropdown(
                                id='predict_habitaciones',
                                options=[{'label': str(h), 'value': h} for h in sorted(df['bedrooms'].dropna().unique())],
                                placeholder="Seleccione el número de habitaciones",
                                style={'backgroundColor': 'white'}
                            )
                        ], style={'marginBottom': '10px'}),
                        html.Div([
                            html.Label("Área en m²:",
                                       style={'color': '#0D47A1'}),
                            dcc.Input(
                                id='predict_area',
                                type='number',
                                placeholder='Ingrese el área en m²',
                                style={'backgroundColor': 'white'}
                            )
                        ], style={'marginBottom': '10px'}),
                        html.Div([
                            html.Label("Ubicación (Ciudad):",
                                       style={'color': '#0D47A1'}),
                            dcc.Dropdown(
                                id='predict_ciudad',
                                options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                                placeholder="Seleccione una ciudad",
                                style={'backgroundColor': 'white'}
                            )
                        ], style={'marginBottom': '10px'}),
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
                        ),
                        html.P("Ingrese las características del inmueble para predecir el precio usando un modelo de Random Forest.",
                               style={'color': '#0D47A1'})
                    ]
                ),
                # Columna de Gráficos (Histograma y Scatter)
                html.Div(
                    style={'width': '65%', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
                    children=[
                        html.Div(
                            style={
                                'width': '48%',
                                'backgroundColor': '#BBDEFB',
                                'padding': '20px',
                                'borderRadius': '5px',
                                'marginBottom': '20px'
                            },
                            children=[
                                html.H3("Filtros para Histograma",
                                        style={'color': '#1565C0'}),
                                html.P("Este gráfico muestra la distribución de los precios de alquiler en función de la ciudad y el precio máximo seleccionado.",
                                       style={'color': '#0D47A1'}),
                                html.Label("Ciudad:",
                                           style={'color': '#0D47A1'}),
                                dcc.Dropdown(
                                    id='cityname_hist',
                                    options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                                    placeholder="Seleccione una ciudad",
                                    style={'backgroundColor': 'white'}
                                ),
                                html.Label("Precio máximo:",
                                           style={'color': '#0D47A1'}),
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
                            style={
                                'width': '48%',
                                'backgroundColor': '#BBDEFB',
                                'padding': '20px',
                                'borderRadius': '5px',
                                'marginBottom': '20px'
                            },
                            children=[
                                html.H3("Filtros para Scatter Plot",
                                        style={'color': '#1565C0'}),
                                html.P("Este gráfico muestra la relación entre el área en m² y el precio, permitiendo filtrar por número de habitaciones y área seleccionada.",
                                       style={'color': '#0D47A1'}),
                                html.Label("Número de habitaciones:",
                                           style={'color': '#0D47A1'}),
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
                # Segunda fila de Gráficos (Heatmaps)
                html.Div(
                    style={'width': '100%', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'},
                    children=[
                        html.Div(
                            style={
                                'width': '48%',
                                'backgroundColor': '#BBDEFB',
                                'padding': '20px',
                                'borderRadius': '5px',
                                'marginBottom': '20px'
                            },
                            children=[
                                html.H3("Filtros para Heatmap de Ubicación",
                                        style={'color': '#1565C0'}),
                                html.P("Este heatmap muestra la distribución geográfica de los precios de alquiler, basándose en la ciudad seleccionada.",
                                       style={'color': '#0D47A1'}),
                                html.Label("Ciudad:",
                                           style={'color': '#0D47A1'}),
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
                            style={
                                'width': '48%',
                                'backgroundColor': '#BBDEFB',
                                'padding': '20px',
                                'borderRadius': '5px',
                                'marginBottom': '20px'
                            },
                            children=[
                                html.H3("Heatmap de Precio vs Variable Elegida",
                                        style={'color': '#1565C0'}),
                                html.P("Este heatmap permite visualizar cómo varía el precio en función de la variable seleccionada, como área, número de habitaciones, latitud o longitud.",
                                       style={'color': '#0D47A1'}),
                                html.Label("Seleccione una variable:",
                                           style={'color': '#0D47A1'}),
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
    ]
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Entrenar modelo de Random Forest
X = df[['bedrooms', 'square_feet']]
y = df['price']
X = X.dropna()
y = y.loc[X.index]  # Asegurar que los datos estén alineados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

@app.callback(
    Output('resultado_prediccion', 'children'),
    [Input('btn_predecir', 'n_clicks')],
    [dash.dependencies.State('predict_habitaciones', 'value'),
     dash.dependencies.State('predict_area', 'value')]
)
def predecir_precio(n_clicks, habitaciones, area):
    if n_clicks > 0 and habitaciones is not None and area is not None:
        import pandas as pd  # Asegurar importación de Pandas
        entrada = pd.DataFrame([[habitaciones, area]], columns=['bedrooms', 'square_feet'])
        prediccion = modelo_rf.predict(entrada)[0]
        prediccion = modelo_rf.predict([[habitaciones, area]])[0]
        return f"Precio estimado: ${prediccion:,.2f}"
    return "Ingrese los valores y presione Predecir Precio"  

# Callbacks para actualizar los gráficos
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
    return px.histogram(df_filtrado, x='price', title='Distribución de Precios')

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
    if df_filtrado.empty:
        return px.scatter(x=[0], y=[0], title='No hay datos disponibles')
    if habitaciones is not None:
        df_filtrado = df_filtrado[df_filtrado['bedrooms'] == habitaciones]
    if df_filtrado.empty:
        return px.scatter(x=[], y=[], title='No hay datos disponibles')
    return px.scatter(df_filtrado, x='square_feet', y='price', title='Relación entre Área y Precio')



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
                              title='Mapa de Calor - Precio por Ubicación',
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
    if df_filtrado.empty:
        return px.density_heatmap(x=[], y=[], title='No hay datos disponibles')
    
    return px.density_heatmap(df_filtrado, x=variable, y='price', z='price',
                              title=f'Mapa de Calor - Precio vs {variable.capitalize()}',
                              labels={variable: variable.capitalize(), 'price': 'Precio'},
                              color_continuous_scale='OrRd')

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)