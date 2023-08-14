import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from utils import data_processing_st

pd.set_option("display.float_format", "{:.1f}".format)

########## File level VARIABLES ##########
##########################################
rfm_output_FILEPATH = "../data/clients.csv"
transactions_aggregated_FILEPATH = "../data/transactions.csv"
##########################################

st.set_page_config(page_title="ClientCo - client churn prediction", layout="wide")

st.header("ClientCo")
st.text("")

################################################################################
st.subheader("Clients about to churn")

c1, c2, c3, c4 = st.columns(4)

n_clients = c1.number_input(label="Number of clients displayed:", value=10)

min_revenue = c2.number_input(label="Minimum of revenue by client:", value=100_000)

fav_channel = c3.multiselect(
    label="Client's favorite channel",
    options=(
        "online",
        "at the store",
        "by phone",
        "during the visit of a sales rep",
        "other",
    ),
)

min_days = c4.number_input(label="Minimum days since last order:", value=7)

data = data_processing_st.preprocess_data(
    rfm_output_FILEPATH, transactions_aggregated_FILEPATH
)
to_display = data_processing_st.get_clients_most_likely_to_churn(
    data, n_clients, min_revenue, fav_channel, min_days
)

st.dataframe(
    to_display[
        [
            "client_id",
            "Churn likelihood (%)",
            "Days since last order",
            "Total revenue (€)",
            "Average basket (€)",
            "Favorite channel",
        ]
    ].style.background_gradient(
        axis=None, cmap="YlOrRd", subset=["Churn likelihood (%)"]
    )
)


################################################################################
st.text("")
st.text("")
st.subheader("Zoom in on a client")

client_id = st.selectbox(
    label="Enter a client ID", options=np.sort(data["client_id"].unique())
)

c1, c2, c3, c4 = st.columns(4)

this_client = data.loc[data["client_id"] == client_id]

c1.metric(
    label="Churn likelihood",
    value=str(this_client["Churn likelihood (%)"].values[0]) + "%",
)

c2.metric(
    label="Days since last order", value=this_client["Days since last order"].values[0]
)

c3.metric(
    label="Average frequency of order (days)",
    value=this_client["Order frequency (days)"],
)

c4.metric(label="Average basket (€)", value=this_client["Average basket (€)"].values[0])

st.text("")
st.text("")

c1, _ = st.columns([4, 1])
c1.pyplot(
    data_processing_st.make_violin_plots(data, client_id, fig_width=10, fig_height=5)
)
