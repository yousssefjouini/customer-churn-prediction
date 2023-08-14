import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def preprocess_data(rfm_output_FILEPATH: str, transactions_aggregated_FILEPATH: str):
    """
    Performs preprocessing on the predictions (rfr_output) and on the transactions.
    """
    # load and merge data
    rfm_output = pd.read_csv(rfm_output_FILEPATH, sep=';')
    transactions_aggregated = pd.read_csv(transactions_aggregated_FILEPATH, sep=';')
    result = rfm_output.merge(transactions_aggregated, on="client_id")

    # rename columns
    result.rename(
        columns={
            "churn_likelihood": "Churn likelihood (%)",
            "sales_net_sum": "Total revenue (€)",
            "sales_net_mean": "Average basket (€)",
            "order_channel_mode": "Favorite channel",
            "days_diff_remove_first_last_mean": "Order frequency (days)",
        },
        inplace=True,
    )

    # fix number formatting
    result["Churn likelihood (%)"] = (result["Churn likelihood (%)"] * 100).astype(int)
    result["Average basket (€)"] = result["Average basket (€)"].round(decimals=1)
    result = result.astype({"Total revenue (€)": "int"})

    # convert date of last order to days delta
    max_date = dt.date(2019, 9, 22)
    result["Days since last order"] = result["date_order_amax"].apply(
        lambda x: (max_date - pd.to_datetime(x).date()).days
    )
    return result


def get_clients_most_likely_to_churn(df, n_clients, min_revenue, fav_channel, min_days):
    """
    Implements the filters on the table displayed in streamlit webapp.
    """
    df.sort_values(by="Churn likelihood (%)", ascending=False, inplace=True)
    if min_revenue is not None:
        df = df.loc[df["Total revenue (€)"] > min_revenue]
    if bool(fav_channel):
        df = df.loc[df["Favorite channel"].isin(fav_channel)]
    if min_days is not None:
        df = df.loc[df["Days since last order"] >= min_days]
    if n_clients is not None:
        df = df.head(n_clients)
    return df


def make_violin_plots(data, client_id, fig_width=10, fig_height=6):
    """
    Creates the plot for comparing a client's RFM output with the distriution of the population.
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax = sns.violinplot(
        data=data[["r_score", "m_score", "f_score"]], palette="pastel", cut=0
    )
    this_client = data.loc[data["client_id"] == client_id]
    this_client = [
        this_client["r_score"].values[0],
        this_client["m_score"].values[0],
        this_client["f_score"].values[0],
    ]
    ax.axhline(
        y=this_client[0],
        xmin=0.0659,
        xmax=0.2659,
        color="red",
        label="Client",
        linewidth=2,
    )
    ax.axhline(y=this_client[1], xmin=0.4, xmax=0.6, color="red", linewidth=2)
    ax.axhline(y=this_client[2], xmin=0.73, xmax=0.93, color="red", linewidth=2)
    ax.legend(loc="upper right")
    ax.set_title("RFM scores for this client vs all clients")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.spines[["right", "top", "bottom"]].set_visible(False)
    return fig
