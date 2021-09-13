import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
import streamlit as st

from idmcomp import IdmComp

#   SEIRV Model for COVID-19 Forecasting Tool:
#   Parameters (based on a paper by Boldog et al (2020):
#   Link to the paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7073711/
#   Also referenced in this Colab notebook:
#   https://colab.research.google.com/drive/1ddb_0swsq9MRKyHrzflCzeF8Tqqmp24H#scrollTo=oGa0Q3yH-Zai


st.set_page_config(page_title='SEIRV Model',
                   page_icon=None,
                   layout='wide',
                   initial_sidebar_state='auto')

#   Title:
st.title('SEIRV Model for COVID-19 Forecasting Tool')
st.write ('(SEIRV: Susceptible, Exposed, Infected, Recovered, Vaccinated)')
st.markdown('---')

col1, col2, col3 = st.columns((1, 1, 2.25))

st.sidebar.header('Initial State Values:')
N_in = st.sidebar.number_input(
    label='N (total population):', min_value=1)

t_duration = col2.number_input(
    label='Simulation duration (in days):',
    value=365)

R0_input = col1.number_input(
    label='Basic Reproduction Number (R0):',
    value=2.4)

t_incubation = col1.number_input(
    label='Incubation period (in days):',
    value=5.1)

t_infection = col1.number_input(
    label='Symptomatic infection period (in days):',
    value=3.3)

v_eff = col2.number_input(
    label='Vaccine efficacy or effectiveness:',
    value=0.6)



mask_use = st.sidebar.radio('100% Strict Mask Use is Observed:',
                            ('Yes', 'No'))
if mask_use == 'Yes':
    R0 = R0_input-(R0_input*0.258)
else:
    R0 = R0_input


model = IdmComp(N=N_in,
                time=t_duration,
                R0=R0,
                t_inc=t_incubation,
                t_inf=t_infection,
                eff=v_eff)


duration, alpha, beta, gamma, c_s = model.idm_rates()

herd_im = round(model.herd_im(), 2)

cov_val = col2.number_input(label='Vaccine coverage:',
                            value=herd_im)

st.markdown('---')
st.markdown('### Vaccine coverage (efficacy = {}%) needed to achieve herd immunity: {}%'.format(v_eff*100, herd_im*100))


E_in = st.sidebar.number_input(
    label='Initial exposed population:', min_value=0)
I_in = st.sidebar.number_input(
    label='Initial infected population:', min_value=0)
R_in = st.sidebar.number_input(
    label='Initial recovered population:', min_value=0)



y_in = model.initial_state_seirv(E_in=1,
                           I_in=0,
                           R_in=0,
                           p=cov_val)

y_in_seir = model.initial_state_seir(E_in=1,
                                     I_in=0,
                                     R_in=0)

## Define the SEIRV model function:
def SEIRV_model(y, time, N, beta, alpha, gamma, c_s):
    S, E, I, R, V = y
    lmbda = beta*I/N

    dS = -lmbda*S
    dE = lmbda*S - alpha*E + c_s*lmbda*V
    dI = alpha*E - gamma*I
    dR = gamma*I
    dV = -c_s*lmbda*V

    return dS, dE, dI, dR, dV



## Solve the ordinary differential equations:
output = odeint(func=SEIRV_model, y0=y_in, t=duration, args=(N_in, beta, alpha, gamma, c_s))
S, E, I, R, V = output.T
output_df = pd.DataFrame(output)
output_df = output_df.rename(columns={0: 'S',
                                      1: 'E',
                                      2: 'I',
                                      3: 'R',
                                      4: 'V'})

output_df.index = output_df.index+1
output_df['Days'] = output_df.index


col3.markdown('#### SEIRV Data Output:')
col3.dataframe(output_df)


## Define the SEIR model function:

def SEIR_model(y, time, N, beta, alpha, gamma):
    S, E, I, R = y
    lmbda = beta*I/N

    dS = -lmbda*S
    dE = lmbda*S - alpha*E
    dI = alpha*E - gamma*I
    dR = gamma*I

    return dS, dE, dI, dR

## Solve the ordinary differential equations:

output2 = odeint(func=SEIR_model, y0=y_in_seir, t=duration, args=(N_in, beta, alpha, gamma))
S, E, I, R = output2.T.round(decimals=0)
output2_df = pd.DataFrame(output2)
output2_df = output2_df.rename(columns={0: 'S',
                                       1: 'E',
                                       2: 'I',
                                       3: 'R'})

output2_df.index = output2_df.index+1
output2_df['Days'] = output2_df.index


st.markdown('---')
st.markdown('### **Simulation Result:**')
with st.expander('See simulation result'):
    # Plot the data:

    model_plot = make_subplots(rows=2, cols=1,
                               subplot_titles=('<b>Simulated COVID-19 figures Using the SEIRV Model:</b><br>'\
                                               '(Vaccine Coverage = {}%, Vaccine Efficacy = {}%)'.format(cov_val*100, v_eff*100),
                                               '<b>Infected Population with no Intervention and With Vaccination = {}% </b><br>'\
                                               '(Vaccine Coverage = {}%, Vaccine Efficacy = {}%)'.format(v_eff*100, cov_val*100, v_eff*100)))



    # First Plot:

    ## Plot of Susceptible (S):
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['S'],
                                    legendgroup=1,
                                    name='Susceptible'),
                         row=1, col=1)

    ## Plot of Exposed (E):
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['E'],
                                    legendgroup=1,
                                    name='Exposed'),
                         row=1, col=1)

    ## Plot of Infected (I):
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['I'],
                                    legendgroup=1,
                                    name='Infected'),
                         row=1, col=1)

    ## Plot of Recovered (R):
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['R'],
                                    legendgroup=1,
                                    name='Recovered'),
                         row=1, col=1)

    ## Plot of Vaccinated (V):
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['V'],
                                    legendgroup=1,
                                    name='Vaccinated ({}% of N)'.format(cov_val*100)),
                         row=1, col=1)



    # Second Plot:

    ## Plot of Exposed with no Intervention:
    model_plot.add_trace(go.Scatter(x=output2_df['Days'],
                                    y=output2_df['E'], fill='tozeroy', mode='lines',
                                    line_color='black', line_width=1,
                                    fillcolor='rgba(0, 230, 64, 0.35)',
                                    legendgroup=2,
                                    name='Unvaccinated Population, Exposed'),
                         row=2, col=1)

    ## Plot of Infected with no Intervention:
    model_plot.add_trace(go.Scatter(x=output2_df['Days'],
                                    y=output2_df['I'], fill='tozeroy', mode='lines',
                                    line_color='black', line_width=1,
                                    fillcolor='rgba(255,0,255, 0.50)',
                                    legendgroup=2,
                                    name='Unvaccinated Population, Infected'),
                         row=2, col=1)


    ## Plot of Exposed with Vaccine:
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['E'], fill='tozeroy', mode='lines',
                                    line_color='black', line_width=1,
                                    fillcolor='rgba(245, 171, 53, 0.5)',
                                    legendgroup=2,
                                    name='Vaccinated Population, Exposed'),
                         row=2, col=1)

    ## Plot of Infected with Vaccine:
    model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                    y=output_df['I'], fill='tozeroy', mode='lines',
                                    line_color='black', line_width=1,
                                    fillcolor='rgba(25, 181, 254, 0.5)',
                                    legendgroup=2,
                                    name='Vaccinated Population, Infected'),
                         row=2, col=1)

    model_plot.update_xaxes(title_text='Time (Days)', row=1, col=1)
    model_plot.update_yaxes(title_text='Population', row=1, col=1)
    model_plot.update_xaxes(title_text='Time (Days)', row=2, col=1)
    model_plot.update_yaxes(title_text='Population', row=2, col=1)

    model_plot.update_layout(xaxis_title='Time (Days)',
                             yaxis_title='Population',
                             legend_tracegroupgap=420,
                             template='seaborn', autosize=False, width=800, height=1000)



    st.plotly_chart(model_plot, use_container_width=True)
