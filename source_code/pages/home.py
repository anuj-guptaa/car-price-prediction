import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/')

layout =  dbc.Container([
    html.H1("Welcome to an assortment of ML/DL projects by Anuj Gupta."),
    html.H1("I hope you enjoy testing out my models!"),

], fluid=True)