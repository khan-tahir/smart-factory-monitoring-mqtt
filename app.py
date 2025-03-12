import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import paho.mqtt.client as mqtt
import joblib

# Load pre-trained predictive model
model = joblib.load("predictive_maintenance_model.pkl")

# Initialize MQTT Client
MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "smart_factory/sensors"

sensor_data = {"time": [], "temperature": [], "vibration": [], "pressure": []}

def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8").split(',')
    sensor_data["time"].append(datetime.now().strftime('%H:%M:%S'))
    sensor_data["temperature"].append(float(payload[0]))
    sensor_data["vibration"].append(float(payload[1]))
    sensor_data["pressure"].append(float(payload[2]))

def setup_mqtt():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_start()
    return client

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Smart Factory Monitoring System", className="text-center my-4"), width=12)]),
    
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Machine Status"),
                          dbc.CardBody([html.H3("Operating Normally", className="text-success", id="machine-status")])]), width=4),
        
        dbc.Col(dbc.Card([dbc.CardHeader("Current Metrics"),
                          dbc.CardBody([
                              html.H4("Temperature: ", className="text-info"), html.H2("-", id="current-temp"),
                              html.H4("Vibration: ", className="text-warning"), html.H2("-", id="current-vib"),
                              html.H4("Pressure: ", className="text-danger"), html.H2("-", id="current-press")])]), width=8)
    ]),
    
    dbc.Row([dbc.Col(dcc.Graph(id='live-graph'), width=12)]),
    
    dcc.Interval(id='graph-update', interval=2000, n_intervals=0)
], fluid=True)

@app.callback(
    [Output('live-graph', 'figure'), Output('current-temp', 'children'),
     Output('current-vib', 'children'), Output('current-press', 'children'),
     Output('machine-status', 'children')],
    [Input('graph-update', 'n_intervals')]
)
def update_graph(n):
    if len(sensor_data['time']) < 10:
        return dash.no_update
    
    df = pd.DataFrame(sensor_data)
    df = df.tail(30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'], name="Temperature (°F)", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['vibration'], name="Vibration (g)", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['pressure'], name="Pressure (PSI)", line=dict(color='blue')))
    
    fig.update_layout(title="Sensor Readings Over Time", xaxis_title="Time", yaxis_title="Value")
    
    latest_data = df.iloc[-1]
    status_prediction = model.predict([[latest_data["temperature"], latest_data["vibration"], latest_data["pressure"]]])
    
    machine_status = "Warning! Maintenance Required" if status_prediction[0] == 1 else "Operating Normally"
    return fig, f"{latest_data['temperature']:.1f}°F", f"{latest_data['vibration']:.2f}g", f"{latest_data['pressure']:.1f} PSI", machine_status

if __name__ == '__main__':
    setup_mqtt()
    app.run_server(debug=True, host='0.0.0.0', port=8050)
