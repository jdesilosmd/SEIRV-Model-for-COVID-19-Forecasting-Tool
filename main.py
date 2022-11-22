import numpy as np                              # Loads the numpy library (for matrix math)
import pandas as pd                             # Loads pandas (for dataframes)
import plotly.express as px                     # Loads plotly express library (for graphs)
import plotly.graph_objects as go               # Loads plotly (for more complex graphs)
from plotly.subplots import make_subplots       # Enables creation of subplots
from scipy.integrate import odeint      # Imports the odeint function from the scipy library to compute for ordinary differential equations
import streamlit as st                          # Loads to streamlit library for the python data dashboard

from idmcomp import IdmComp             # Loads the IdmComp to compute for the SEIR and SEIRV model (COVID-19, measles, etc.)
from idmcomp import DengueComp          # Loads the DengueComp to compute for the Ross-Macdonald model (Dengue)

#   Compartment models for infectious diseases forecasting:
#   COVID-19 Parameters (based on a paper by Boldog et al (2020))
#   Link to the paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7073711/
#   Ross Macdonald parameters (for dengue): https://doi.org/10.1007%2Fs11538-020-00723-0
#   Models are based on the Imperial College London's course on infectious disease modeling
#   Also referenced in this Colab notebook:
#   https://colab.research.google.com/drive/1ddb_0swsq9MRKyHrzflCzeF8Tqqmp24H#scrollTo=oGa0Q3yH-Zai
#   SEIRV = Susceptible-Exposed-Infected-Recovered-Vaccinated
#   SEIR = Susceptible-Exposed-Infected-Recovered


st.set_page_config(page_title='Compartment Model',      # Sets the page title
                   page_icon=None,                      # Can be used to place a page icon
                   layout='wide',                       # Sets the default page layout to wide
                   initial_sidebar_state='auto')        # Sidebar setting

#   Title:
st.title('Compartment models for infectious diseases in a hospital or LGU setting')
st.write('Two Models:')
st.write('Model 1 for Coronavirus, Influenza, or Measles')
st.write('Model 2 for Dengue')
st.markdown('---')


# Creates 2 tabs for Model 1 and 2
tab1, tab2 = st.tabs(["Model 1", "Model 2"])

# Code for Tab 1
with tab1:

    # Create header
    st.header('Infectious Disease Model for Coronavirus, Influenza, or Measles')

    # Write text using markdown
    st.markdown('#### Initial State Values:')

    # Create 3 columns with gap set to "large"
    col1, col2, col3 = st.columns(3, gap="large")

    #create number inputs for each of the parameters
    N_in = col1.number_input(
        label='N (total population):', min_value=1)
    t_duration = col1.number_input(
        label='Simulation duration (in days):',
        value=365)

    R0_input = col1.number_input(
        label='Basic Reproduction Number (R0):',
        value=2.4)

    t_incubation = col2.number_input(
        label='Incubation period (in days):',
        value=5.1)

    t_infection = col2.number_input(
        label='Symptomatic infection period (in days):',
        value=3.3)


    # Page break
    st.markdown('---')


    # Create another 3 columns with gap set to "large"
    col4, col5, col6 = st.columns(3, gap="large")

    # create number inputs for each of the parameters

    with col4:
        E_in = st.number_input(
        label='Initial exposed population:', min_value=0)
    with col5:
        I_in = st.number_input(
        label='Initial infected population:', min_value=0)
    with col6:
        R_in = st.number_input(
        label='Initial recovered population:', min_value=0)

    st.markdown('---')


    #Create 2 tabs for the "SEIRV model" and "SEIR model"
    tab3, tab4 = st.tabs(["SEIRV Model", "SEIR Model"])

    with tab3:

        col7, col8 = st.columns([1, 2], gap="large")

        with col7:

            v_eff = st.number_input(
                label='Vaccine efficacy or effectiveness:',
                value=0.6)

            mask_use = st.radio('100% Strict Mask Use:',
                                  ('Yes', 'No'))
            if mask_use == 'Yes':
                R0 = R0_input - (R0_input * 0.258)
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

            cov_val = st.number_input(label='Vaccine coverage:',
                                        value=herd_im)

            st.markdown(
                '###### Vaccine coverage (efficacy = {}%) needed to achieve herd immunity: {}%'.format(v_eff * 100,
                                                                                                      herd_im * 100))
            y_in = model.initial_state_seirv(E_in=E_in,
                                       I_in=I_in,
                                       R_in=R_in,
                                       p=cov_val)


            ## Define the SEIRV model function:
            def SEIRV_model(y, N, time, beta, alpha, gamma, c_s):
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
            output_df = output_df.apply(np.ceil)

            st.markdown('##### SEIRV Data Output:')
            st.dataframe(output_df, height=300)


        with col8:
            st.markdown('##### **Simulation Result:**')
            model_plot = go.Figure()

            ## Plot of Susceptible (S):
            model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                            y=output_df['S'],
                                            legendgroup=1,
                                            name='Susceptible'))

            ## Plot of Exposed (E):
            model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                            y=output_df['E'],
                                            legendgroup=1,
                                            name='Exposed'))

            ## Plot of Infected (I):
            model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                            y=output_df['I'],
                                            legendgroup=1,
                                            name='Infected'))

            ## Plot of Recovered (R):
            model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                            y=output_df['R'],
                                            legendgroup=1,
                                            name='Recovered'))

            ## Plot of Vaccinated (V):
            model_plot.add_trace(go.Scatter(x=output_df['Days'],
                                            y=output_df['V'],
                                            legendgroup=1,
                                            name='Vaccinated ({}% of N)'.format(cov_val)))

            ## Update graph layout
            model_plot.update_layout(
                title={
                    'text': '<b>SEIRV Model Projection:</b><br>' \
                            '(Vaccine Coverage = {}%, Vaccine Efficacy = {}%)'.format(cov_val * 100, v_eff * 100)
                }
            )
            model_plot.update_xaxes(title_text='Time (Days)')
            model_plot.update_yaxes(title_text='Population')
            model_plot.update_xaxes(title_text='Time (Days)')
            model_plot.update_yaxes(title_text='Population')

            model_plot.update_layout(xaxis_title='Time (Days)',
                                     yaxis_title='Population',
                                     legend_tracegroupgap=420,
                                     template='seaborn', autosize=False, width=600, height=700)

            ## Displays the plotly chart in streamlit
            st.plotly_chart(model_plot, use_container_width=True)

    # Tab 4
    with tab4:

        col9, col10 = st.columns([1, 2], gap="large")

        with col9:

            y_in_seir = model.initial_state_seir(E_in=E_in,
                                                 I_in=I_in,
                                                 R_in=R_in)



            ## Define the SEIR model function:

            def SEIR_model(y, N, time, beta, alpha, gamma):
                S, E, I, R = y
                lmbda = beta*I/N

                dS = -lmbda*S
                dE = lmbda*S - alpha*E
                dI = alpha*E - gamma*I
                dR = gamma*I

                return dS, dE, dI, dR

            ## Solve the ordinary differential equations:

            output2 = odeint(func=SEIR_model, y0=y_in_seir, t=duration, args=(N_in, beta, alpha, gamma))
            S, E, I, R = output2.T
            output2_df = pd.DataFrame(output2)
            output2_df = output2_df.rename(columns={0: 'S',
                                                   1: 'E',
                                                   2: 'I',
                                                   3: 'R'})

            output2_df.index = output2_df.index+1
            output2_df['Days'] = output2_df.index
            output2_df = output2_df.apply(np.ceil)

            st.markdown('##### SEIR Data Output:')
            st.dataframe(output2_df, height=800)

        # Column 10
        with col10:

            st.markdown('##### **Simulation Result:**')
            model_plot2 = go.Figure()

            ## Plot of Susceptible (S):
            model_plot2.add_trace(go.Scatter(x=output2_df['Days'],
                                            y=output2_df['S'],
                                            legendgroup=1,
                                            name='Susceptible'))

            ## Plot of Exposed (E):
            model_plot2.add_trace(go.Scatter(x=output2_df['Days'],
                                            y=output2_df['E'],
                                            legendgroup=1,
                                            name='Exposed'))

            ## Plot of Infected (I):
            model_plot2.add_trace(go.Scatter(x=output2_df['Days'],
                                            y=output2_df['I'],
                                            legendgroup=1,
                                            name='Infected'))

            ## Plot of Recovered (R):
            model_plot2.add_trace(go.Scatter(x=output2_df['Days'],
                                            y=output2_df['R'],
                                            legendgroup=1,
                                            name='Recovered'))

            ## Update graph layout
            model_plot2.update_layout(
                title={
                    'text': '<b>SEIR Model Projection:</b><br>' \
                            '(Zero or Negligible Vaccine Coverage)'
                }
            )
            model_plot2.update_xaxes(title_text='Time (Days)')
            model_plot2.update_yaxes(title_text='Population')
            model_plot2.update_xaxes(title_text='Time (Days)')
            model_plot2.update_yaxes(title_text='Population')

            model_plot2.update_layout(xaxis_title='Time (Days)',
                                     yaxis_title='Population',
                                     legend_tracegroupgap=420,
                                     template='seaborn', autosize=False, width=600, height=700)

            ## Displays the plotly chart in streamlit
            st.plotly_chart(model_plot2, use_container_width=True)



    # Tab 2
    with tab2:

        st.header('Infectious Disease Model for Dengue')
        st.markdown('#### Initial State Values:')

        # Create 4 columns with default gap set to "large"
        col6, col7, col8, col9 = st.columns(4, gap="large")

        # create number inputs for each of the parameters
        N_h = col6.number_input(
            label='N (host population):', value=10000)

        N_v = col6.number_input(
            label='N (vector population):', value=N_h*2)

        t_duration = col7.number_input(
            label='Simulation duration (in days):',
            value=180)

        bite_n = col7.number_input(
            label='Number of hosts a mosquito bite in a day:',
            value=1)

        bv_input = col8.number_input(
            label='Probability of infection (host to vector):',
            value=0.4)

        bh_input = col8.number_input(
            label='Probability of infection (vector to host):',
            value=0.4)

        uv_input = col9.number_input(
            label='Vector mortality rate:',
            value=0.25)

        h_recov_input = col9.number_input(
            label='Recovery rate from dengue in humans:',
            value=0.167)

        # Create model 2 (for dengue) using the DengueComp class from IdmComp
        model2 = DengueComp(
            Nv = N_v,
            Nh = N_h,
            time = t_duration,
            t_bite = bite_n,
            bv = bv_input,
            bh = bh_input,
            uv = uv_input,
            h_recov = h_recov_input)

        # assign values to the variables below using the dengue_rates function from model2
        duration, bite_rate, bv, bh, uv, h_recov = model2.dengue_rates()

        st.markdown('---')

        col10, col11, col12 = st.columns(3, gap="large")


        ### Initial State Inputs:

        Ih_in = col10.number_input(
            label='Initial infected host population:', min_value=0)
        Rh_in = col11.number_input(
            label='Initial recovered host population:', min_value=0)
        Iv_in = col12.number_input(
            label='Initial infected vector population:', min_value=0)

        st.markdown('---')



        y_in_dengue = model2.initial_state_dengue(Ih_in=Ih_in,
                                                  Rh_in=Rh_in,
                                                  Iv_in=Iv_in)


        ## Define the Ross MacDonald function:

        def RM_model(y, N_h, N_v, time, bite_rate, bv, bh, uv, h_recov):
            Sh, Ih, Rh, Sv, Iv = y

            ## Total host and vector population:
            N_h = Sh+Ih+Rh
            N_v = Sv+Iv

            ## Host calculation
            dSh = -bite_rate*bh*Sh*Iv/N_h
            dIh = (bite_rate*bh*Sh*Iv/N_h)-h_recov*Ih
            dRh = h_recov*Ih

            ## Vector calculation
            dSv = uv*N_v-(bite_rate*bv*Sv*Ih/N_h)-uv*Sv
            dIv = ((bite_rate*bv/N_h)*Sv*Ih)-uv*Iv

            return dSh, dIh, dRh, dSv, dIv


        ## Solve the ordinary differential equations:

        output3 = odeint(func=RM_model, y0=y_in_dengue, t=duration, args=(N_h, N_v, bite_rate, bv, bh, uv, h_recov))
        Sh, Ih, Rh, Sv, Iv = output3.T
        output3_df = pd.DataFrame(output3)
        output3_df = output3_df.rename(columns={0: 'Sh',
                                                1: 'Ih',
                                                2: 'Rh',
                                                3: 'Sv',
                                                4: 'Iv'})

        output3_df.index = output3_df.index + 1
        output3_df['Days'] = output3_df.index
        output3_df = output3_df.apply(np.ceil)

        col13, col14 = st.columns([1, 2], gap="large")

        with col13:
            st.markdown('##### SEIR Data Output:')
            st.dataframe(output3_df, height=500)


        with col14:
            st.markdown('##### **Simulation Result:**')
            model_plot3 = go.Figure()

            ## Plot of Susceptible host (Sh):
            model_plot3.add_trace(go.Scatter(x=output3_df['Days'],
                                             y=output3_df['Sh'],
                                             legendgroup=1,
                                             name='Susceptible Host'))

            ## Plot of Infected host (Ih):
            model_plot3.add_trace(go.Scatter(x=output3_df['Days'],
                                             y=output3_df['Ih'],
                                             legendgroup=1,
                                             name='Infected Host'))

            ## Plot of Recovered (Rh):
            model_plot3.add_trace(go.Scatter(x=output3_df['Days'],
                                             y=output3_df['Rh'],
                                             legendgroup=1,
                                             name='Recovered Host'))

            ## Plot of Susceptible vector (Sv):
            model_plot3.add_trace(go.Scatter(x=output3_df['Days'],
                                             y=output3_df['Sv'],
                                             legendgroup=1,
                                             name='Susceptible Vector'))

            ## Plot of Infected vector (Iv):
            model_plot3.add_trace(go.Scatter(x=output3_df['Days'],
                                             y=output3_df['Iv'],
                                             legendgroup=1,
                                             name='Infected Vector'))

            ## Update graph layout
            model_plot3.update_layout(
                title={
                    'text': '<b>Ross Macdonald Model Projection for Dengue:</b><br>' \
                            '(Vector = <i>Aedes aegypti</i> Mosquito)'
                }
            )
            model_plot3.update_xaxes(title_text='Time (Days)')
            model_plot3.update_yaxes(title_text='Population')
            model_plot3.update_xaxes(title_text='Time (Days)')
            model_plot3.update_yaxes(title_text='Population')

            model_plot3.update_layout(xaxis_title='Time (Days)',
                                     yaxis_title='Population',
                                     legend_tracegroupgap=420,
                                     template='seaborn', autosize=False, width=600, height=500)

            ## Displays the plotly chart in streamlit
            st.plotly_chart(model_plot3, use_container_width=True)
