import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Load Data
df1 = pd.read_csv('dataframe_model.csv')
df3 = pd.read_csv("dataset_plots_disposets_year.csv")

#style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#dash app
app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#plots

## plot1 disposets per year
fig1 = px.bar(df3, x= "year", y= "htn-disposet")

## plot2 superdisposets year
fig2= px.bar(df3, x= "year", y= "sds-psg")

##plot3 number occurrences per disposets
# attention: after filtering materialbelegs out with positive quantities and taking out
## the disposets that are from corona pandemic period (01.03.20-31.12.2021) and that have less
# than 10 movements on the material data
fig3= px.histogram(df1.sort_values(by="number_occurrences", ascending=False), x="disposets", y= "number_occurrences")



## plot4: dispersion analysis per year
years=['2017', '2018', '2019', "2020", "2021", "2022"]

fig4 = go.Figure(data=[
    go.Bar(name='underdispersion', x=years, y=[350, 384, 378, 220, 204, 268]),
    go.Bar(name='acceptance model', x= years, y=[323, 358, 347, 146, 157,215]),
    go.Bar(name='overdispersion', x= years, y=[1155, 1136, 1169, 1275, 1076, 1179])
    
])
# Change the bar mode
fig4.update_layout(barmode='group')
#fig4.show()

## plot5: distribution number occurrences plotly
hist_data = [list(df1["number_occurrences"])]
group_labels = ['number_occurrences'] # name of the dataset
fig5 = ff.create_distplot(hist_data, group_labels)
#fig5.show()

#plot6: maxim_service_disposet vs number occurrences global

fig6 = px.histogram(df1, x="maxim_service_disposet", y="number_occurrences", #color="sex",
                   marginal="box", # or violin, rug
                   hover_data=df1.columns)
#fig6.show()

# plot7: boxplot maximum service disposet
fig7 = px.box(df1, x="maxim_service_disposet")
#fig.show()

# plot 8: maxim service disposet, number occurrences per disposet classification
fig8 = px.histogram(df1, x="maxim_service_disposet", y="number_occurrences", color="disposet_classification",
                   marginal="box", # or violin, rug
                   hover_data=df1.columns)
#fig.show()


#plot9: densities of the maximum service disposet per dispersion class
# Add histogram data
x1 = df1[df1["disposet_classification"]=="overdispersion"]["maxim_service_disposet"].astype(np.float64)
x2 = df1[df1["disposet_classification"]=="underdispersion"]["maxim_service_disposet"].astype(np.float64)
x3 = df1[df1["disposet_classification"]=="acceptance model"]["maxim_service_disposet"].astype(np.float64)


# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Overdispersion', 'Underdispersion', 'Acceptance model']

# Create distplot with custom bin_size
fig9 = ff.create_distplot(hist_data, group_labels, bin_size=50)
#fig9.show()

#plot10: boxplot of the maximum service disposet per class dispersion
x1 = df1[df1["disposet_classification"]=="overdispersion"]["maxim_service_disposet"]
x2 = df1[df1["disposet_classification"]=="underdispersion"]["maxim_service_disposet"]
x3 = df1[df1["disposet_classification"]=="acceptance model"]["maxim_service_disposet"]
fig10 = go.Figure()
fig10.add_trace(go.Box(y=x1, name="overdispersion"))
fig10.add_trace(go.Box(y=x2, name="underdispersion"))
fig10.add_trace(go.Box(y=x3, name="acceptance model"))
fig10.update_traces(boxpoints='all', jitter=0)
#fig10.show()

#plot11: relatisonship between the maximum time service and number occurrences per dispersion class
fig11 = px.scatter(df1, x="number_occurrences", y="maxim_service_disposet", color="disposet_classification",
    size="number_occurrences", size_max=45, log_x=True)
fig11.update_layout(legend=dict(
    yanchor="bottom",
    y=0.99,
    xanchor="right",
    x=0.01
))
#fig11.show()



# plot12: class stocks on the dataset
fig12 = px.histogram(df1, y = "stock")

#plot 13: lambda rate
df2=df1.groupby("sds-psg_x")["lambda_rate"].mean().to_frame().reset_index().sort_values(by="lambda_rate", ascending=False)
fig13= px.histogram(df2, x= "sds-psg_x", y="lambda_rate")


#plot14: 
fig14 = px.histogram(df1, x="maxim_service_disposet", y="number_occurrences", color="stock",
                   marginal="box", # or violin, rug
                   hover_data=df1.columns)
#fig14.show()

#plot15: densities of the disposet maximum service feature on the stock class
# Add histogram data
x1 = np.array(df1[df1["stock"]=="understock"]["maxim_service_disposet"]).astype(np.float64)
x2 = np.array(df1[df1["stock"]=="overstock"]["maxim_service_disposet"]).astype(np.float64)
x3 = np.array(df1[df1["stock"]=="stock model accepted"]["maxim_service_disposet"]).astype(np.float64)
# Group data together
hist_data = [x1, x2, x3]
group_labels = ['understock', 'overstock', 'stock model accepted']
# Create distplot with custom bin_size
fig15 = ff.create_distplot(hist_data, group_labels, bin_size=50)
# fig15.show()

#plot16: boxplot of the disposet maximum time of service per stock class
x1 = np.array(df1[df1["stock"]=="understock"]["maxim_service_disposet"]).astype(np.float64)
x2 = np.array(df1[df1["stock"]=="overstock"]["maxim_service_disposet"]).astype(np.float64)
x3 = np.array(df1[df1["stock"]=="stock model accepted"]["maxim_service_disposet"]).astype(np.float64)
fig16 = go.Figure()
fig16.add_trace(go.Box(y=x1, name="understock"))
fig16.add_trace(go.Box(y=x2, name="overstock"))
fig16.add_trace(go.Box(y=x3, name="stock model accepted"))
fig16.update_traces(boxpoints='all', jitter=0)
#fig16.show()


# plot17: boxplot all together
fig17 = px.box(df1, y="maxim_service_disposet", x="stock", color="disposet_classification")
fig17.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
#fig17.show()

app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='Distribution of number of disposets per year'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph1',
            figure=fig1
        ),  
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Distribution of the number of super-disposets per year.'),

        html.Div(children='''
             
        '''),

        dcc.Graph(
            id='graph2',
            figure=fig2
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Number of material movements per disposet'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph3',
            figure=fig3
        ),  
    ]),




    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Conclusion of the distribution analysis per year'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph4',
            figure=fig4
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Distribution of the number of material movements'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph5',
            figure=fig5
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Histogram of number of material movements on the disposet maximum time of service feature'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph6',
            figure=fig6
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Inter-quantiles of the maximum time of service feature'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph7',
            figure=fig7
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Maximum time of service on the distribution classes'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph8',
            figure=fig8
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Density of the maximum time of service per disposet on the distribution classes'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph9',
            figure=fig9
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Inter-quantiles of the maximum time of service of a disposet on the distribution classes'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph10',
            figure=fig10
        ),  
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Maximum time of service and material movements per dispersion class'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph11',
            figure=fig11
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Frequencies of the stock group (disposet level)'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph12',
            figure=fig12
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='The lambda rate histogram'),

        html.Div(children='''
            
        '''),

        dcc.Graph(
            id='graph13',
            figure=fig13
        ),  
    ]),

    # New Div for all elements in the new 'row' of the page
    #html.Div([
     #   html.H1(children='The disposet maximum time of service and material movements on the stock groups'),

      #  html.Div(children='''
       #     Few comments.
       # '''),

       # dcc.Graph(
       #     id='graph14',
       #     figure=fig14
       # ),  
   # ]),

    # New Div for all elements in the new 'row' of the page
    #html.Div([
     #   html.H1(children='Densities of the disposet maximum service feature for each stock class'),

      #  html.Div(children='''
      #      Few comments.
      #  '''),

       # dcc.Graph(
       #     id='graph15',
       #     figure=fig15
       # ),  
    #]),
    # New Div for all elements in the new 'row' of the page
    #html.Div([
     #   html.H1(children='Boxplot of the disposet maximum time of service per stock class'),

      #  html.Div(children='''
       #     Few comments.
       # '''),

       # dcc.Graph(
       #     id='graph16',
       #     figure=fig16
       # ),  
    #]),
    # New Div for all elements in the new 'row' of the page
    #html.Div([
     #   html.H1(children='Inter-quantiles for the disposet maximum service for different classes'),

      #  html.Div(children='''
       #     Few comments.
       # '''),

       # dcc.Graph(
       #     id='graph17',
       #     figure=fig17
       # ),  
    #]),

    
])







if __name__ == '__main__':
       app.run_server(debug=True)