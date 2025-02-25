import dash
from dash import dcc  # dash core components
from dash import html  # dash html components
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Cargar datos
file_path = "./x_test.csv"
df = pd.read_csv(file_path)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Asegurar que la columna de precios existe y es consistente
df.rename(columns={'real_price': 'price'}, inplace=True)

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Diseño del tablero
app.layout = html.Div([
    # Sección de Predicción con Random Forest
    html.Div([
        html.H3("Predicción del Precio con Random Forest"),
        html.Div([
            html.Label("Número de habitaciones:"),
            dcc.Dropdown(id='predict_habitaciones', 
                         options=[{'label': str(h), 'value': h} for h in sorted(df['bedrooms'].dropna().unique())],
                         placeholder="Seleccione el número de habitaciones")
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label("Área en m²:"),
            dcc.Input(id='predict_area', type='number', placeholder='Ingrese el área en m²')
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label("Ubicación (Ciudad):"),
            dcc.Dropdown(id='predict_ciudad', 
                         options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                         placeholder="Seleccione una ciudad")
        ], style={'margin-bottom': '10px'}),
        html.Button('Predecir Precio', id='btn_predecir', n_clicks=0, style={'margin-top': '10px'}),
        html.Div(id='resultado_prediccion', style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': 'bold'}),
        html.P("Ingrese las características del inmueble para predecir el precio usando un modelo de Random Forest.")
        
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
    html.H1("Tablero de Datos - Alquiler de Apartamentos"),

    # Filtros para Histograma
    html.Div([
        html.H3("Filtros para Histograma"),
        html.P("Este gráfico muestra la distribución de los precios de alquiler en función de la ciudad y el precio máximo seleccionado."),
        html.Label("Ciudad:"),
        dcc.Dropdown(id='cityname_hist', 
                     options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                     placeholder="Seleccione una ciudad"),
        html.Label("Precio máximo:"),
        dcc.Dropdown(id='precio_max_hist', 
                     options=[{'label': f"${p:,.0f}", 'value': p} for p in sorted(df['price'].dropna().unique())],
                     placeholder="Seleccione el precio máximo"),
        dcc.Graph(id='histograma_precios')
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

    # Filtros para Scatter Plot
    html.Div([
        html.H3("Filtros para Scatter Plot"),
        html.P("Este gráfico muestra la relación entre el área en m² y el precio, permitiendo filtrar por número de habitaciones y área seleccionada."),
        html.Label("Número de habitaciones:"),
        dcc.Dropdown(id='habitaciones_scatter', 
                     options=[{'label': str(h), 'value': h} for h in sorted(df['bedrooms'].dropna().unique())],
                     placeholder="Seleccione el número de habitaciones"),
        html.Label("Área en m²:"),
        dcc.Dropdown(id='area_scatter', 
                     options=[{'label': str(a), 'value': a} for a in sorted(df['square_feet'].dropna().unique())],
                     placeholder="Seleccione el área en m²"),
        dcc.Graph(id='scatter_precio_area')
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

    # Filtros para Heatmap de Ubicación
    html.Div([
        html.H3("Filtros para Heatmap de Ubicación"),
        html.P("Este heatmap muestra la distribución geográfica de los precios de alquiler, basándose en la ciudad seleccionada."),
        html.Label("Ciudad:"),
        dcc.Dropdown(id='cityname_heatmap', 
                     options=[{'label': loc, 'value': loc} for loc in df['cityname'].dropna().unique()],
                     placeholder="Seleccione una ciudad"),
        dcc.Graph(id='heatmap_precio_ubicacion')
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

    # Filtros para Heatmap de Precio con Variable Elegida
    html.Div([
        html.H3("Heatmap de Precio vs Variable Elegida"),
        html.P("Este heatmap permite visualizar cómo varía el precio en función de la variable seleccionada, como área, número de habitaciones, latitud o longitud."),
        html.Label("Seleccione una variable:"),
        dcc.Dropdown(id='heatmap_variable',
                     options=[{'label': 'Área (m²)', 'value': 'square_feet'},
                              {'label': 'Número de habitaciones', 'value': 'bedrooms'},
                              {'label': 'Latitud', 'value': 'latitude'},
                              {'label': 'Longitud', 'value': 'longitude'}],
                     placeholder="Seleccione una variable"),
        dcc.Graph(id='heatmap_precio_variable')
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
])

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
    [Input('habitaciones_scatter', 'value'),
     Input('area_scatter', 'value')]
)
def actualizar_scatter(habitaciones, area):
    df_filtrado = df.copy()
    if df_filtrado.empty:
        return px.scatter(x=[0], y=[0], title='No hay datos disponibles')
    if habitaciones is not None:
        df_filtrado = df_filtrado[df_filtrado['bedrooms'] == habitaciones]
    if area is not None:
        df_filtrado = df_filtrado[df_filtrado['square_feet'] == area]
    if df_filtrado.empty:
        return px.scatter(x=[0], y=[0], title='No hay datos disponibles')
    if df_filtrado.empty:
        return px.scatter(x=[0], y=[0], title='No hay datos disponibles')
    if habitaciones is not None:
        df_filtrado = df_filtrado[df_filtrado['bedrooms'] == habitaciones]
    if area is not None:
        df_filtrado = df_filtrado[df_filtrado['square_feet'] == area]
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
