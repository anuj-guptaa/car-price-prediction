import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/model1')


# Creating FORM
name = html.Div(
    [
        dbc.Label("Manufacturer", html_for="example-email"),
        dbc.Input(id="name", type="text", placeholder="Input manufucturer of car"),
        dbc.FormText(
            "This is the value for the car manufacturer",
            color="secondary",
        ),
    ],
    className="mb-3",
)

km_driven = html.Div(
    [
        dbc.Label("KM Driven", html_for="example-email"),
        dbc.Input(id="km_driven", type="number", placeholder="Input kilometers driven of the car"),
        dbc.FormText(
            "This is the value for km_driven",
            color="secondary",
        ),
    ],
    className="mb-3",
)

fuel = html.Div(
    [
        dbc.Label("Fuel Type", html_for="example-email"),
        dbc.RadioItems(
          id="fuel",
          inline=True,
          options=[
            {'label': 'Diesel', 'value': 0},
            {'label': 'Petrol', 'value': 1},
          ],
        ),
    ],
    className="mb-3",
)

seller_type = html.Div(
    [
        dbc.Label("Seller Type", html_for="example-email"),
        dbc.RadioItems(
          id="seller_type",
          inline=True,
          options=[
            {'label': 'Dealer', 'value': "dealer"},
            {'label': 'Individual', 'value': "individual"},
            {'label': 'Trustmark', 'value': "trustmark"}
          ],
        ),
    ],
    className="mb-3",
)

transmission = html.Div(
    [
        dbc.Label("Transmission Type", html_for="example-email"),
        dbc.RadioItems(
          id="transmission",
          inline=True,
          options=[
            {'label': 'Automatic', 'value': 0},
            {'label': 'Manual', 'value': 1}
          ],
        ),
    ],
    className="mb-3",
)

owner = html.Div(
    [
        dbc.Label("Owner Type", html_for="example-email"),
        dbc.RadioItems(
          id="owner",
          inline=True,
          options=[
            {'label': 'First owner', 'value': 1},
            {'label': 'Second owner', 'value': 2},
            {'label': 'Third owner', 'value': 3},
            {'label': 'Fourth or higher owner', 'value': 4},
            {'label': 'Test driven car', 'value': 5},
          ],
        ),
    ],
    className="mb-3",
)

mileage = html.Div(
    [
        dbc.Label("Mileage (kmpl)", html_for="example-email"),
        dbc.Input(id="mileage", type="number", placeholder="Input mileage of the car in kmpl"),
        dbc.FormText(
            "This is the value for mileage",
            color="secondary",
        ),
    ],
    className="mb-3",
)

engine = html.Div(
    [
        dbc.Label("Engine (CC)", html_for="example-email"),
        dbc.Input(id="engine", type="number", placeholder="Input engine of the car in CC"),
        dbc.FormText(
            "This is the value for engine",
            color="secondary",
        ),
    ],
    className="mb-3",
)

max_power = html.Div(
    [
        dbc.Label("Max Power (bhp)", html_for="example-email"),
        dbc.Input(id="max_power", type="number", placeholder="Input max power of the car in bhp"),
        dbc.FormText(
            "This is the value for max_power",
            color="secondary",
        ),
    ],
    className="mb-3",
)

seats = html.Div(
    [
        dbc.Label("Seats", html_for="example-email"),
        dbc.Input(id="seats", type="number", placeholder="Input seats of the car in bhp"),
        dbc.FormText(
            "This is the value for seats",
            color="secondary",
        ),
    ],
    className="mb-3",
)

submit_hardcode = html.Div([
            dbc.Button(id="submit", children="calculate selling price using hardcode", color="primary", className="me-1"),
            dbc.Label("y is: "),
            html.Output(id="y", children="")
], style={'marginTop':'10px'})

submit_model = html.Div([
            dbc.Button(id="submit_model", children="calculate selling price using model", color="primary", className="me-1"),
            dbc.Label("y is: "),
            html.Output(id="y_model", children="")
], style={'marginTop':'10px'})

form =  dbc.Form([
            name,
            km_driven,
            fuel,
            seller_type,
            transmission,
            owner,
            mileage,
            engine,
            max_power,
            seats,
            submit_hardcode,
            submit_model
        ],
        className="mb-3")


# Explain Text
text = html.Div([
    html.H1("A1 - Car Price Prediction"),
    html.P("Please input custom data for your car and receive a predicted selling price."),
    html.P("The model uses a random forest regressor."),
    html.P(""),
])

# Dataset Example
from dash import Dash, dash_table
import pandas as pd
df = pd.read_csv(r'/root/code/datasets/Cars_abridged.csv')

table = dbc.Table.from_dataframe(df, 
                        striped=True, 
                        bordered=True, 
                        hover=True,
                        responsive=True,
                        size='sm'
                            )

layout =  dbc.Container([
        text,
        form,
        html.H1("The Dataset I use to train the model (Only the first few rows displayed for faster loading)"),
        table
    ], fluid=True)

# Accessing user inputs and loading model
import pickle
import sklearn
import pandas

with open('models/random_forest_regressor.pickle', 'rb') as f:
  model = pickle.load(f)

columns = ['km_driven', 'fuel', 'transmission', 'owner', 'mileage',
       'engine', 'max_power', 'seats', 'seller_type_Dealer',
       'seller_type_Individual', 'seller_type_Trustmark Dealer', 'Audi',
       'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat',
       'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu',
       'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus',
       'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz',
       'Mitsubishi', 'Nissan', 'Opel', 'Peugeot',
       'Renault', 'Skoda', 'Tata', 'Toyota',
       'Volkswagen', 'Volvo']

X_input = pd.DataFrame(columns=columns)

name,km_driven,fuel,transmission,,mileage,engine,max_power,seats,'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer'
'Maruti', 145500, 'Diesel',1, 1, 23.4, 1248, 74, 5, 0, 1, 0


@callback(
    Output(component_id="y", component_property="children"),
    State(component_id="name", component_property="value"),
    State(component_id="km_driven", component_property="value"),
    State(component_id="fuel", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def calculate_y_hardcode(name, km_driven, fuel, submit):
    
    return name

# @callback(
#     Output(component_id="y_model", component_property="children"),
#     State(component_id="x_1", component_property="value"),
#     State(component_id="x_2", component_property="value"),
#     Input(component_id="submit_model", component_property='n_clicks'),
#     prevent_initial_call=True
# )
# def calculate_y_model(x_1, x_2, submit):
#     from utils import load
#     import pandas as pd
#     import numpy as np
#     model = load('./models/myModel.pickle')
#     X = np.array([x_1,x_2]).reshape(-1,2)
#     X = pd.DataFrame(X, columns=['x1', 'x2']) 
#     pred = model.predict(X)
#     return f" model said: {pred=} {model.coef_=}"
