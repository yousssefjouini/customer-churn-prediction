import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import plotly.express as px

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


class Preprocessor:
    def __init__(self, file_path: str, nrows=None) -> None:
        if file_path:
            if nrows:
                self.df = pd.read_csv(file_path, sep=";", nrows=nrows)
            else:
                self.df = pd.read_csv(file_path, sep=";")
        else:
            self.df = None
        self.rfm = None
        self.preprocessed_df = None
        self.model = None

    def unique_counts(self):
        for i in self.df.columns:
            count = self.df[i].nunique()
            print(i, ": ", count)

    def clean_dates(self):
        self.df["date_order"] = pd.to_datetime(self.df["date_order"], format="%Y-%m-%d")
        self.df["date_invoice"] = pd.to_datetime(
            self.df["date_invoice"], format="%Y-%m-%d"
        )
        self.df["year"] = self.df["date_order"].dt.year
        self.df["month"] = self.df["date_order"].dt.month
        self.df["difference_invoice_order"] = (
            self.df["date_invoice"] - self.df["date_order"]
        ).dt.days

    def add_sales_tot(self):
        self.df["sales_tot"] = self.df["sales_net"] * self.df["quantity"]

    def drop_null_rows(self):
        self.df = self.df.dropna().reset_index(drop=True)

    def remove_negative_rows(self):
        self.df = self.df[self.df["sales_net"] > 0].dropna().reset_index(drop=True)

    def full_preprocessing(self):
        print("cleaning dates, adding year and month...")
        self.clean_dates()
        print("adding total sales...")
        self.add_sales_tot()
        print("dropping null rows...")
        self.drop_null_rows()
        print("removing negative rows...")
        self.remove_negative_rows()

    def add_recency(self) -> pd.DataFrame:
        recency = self.df.groupby(by=["client_id"])[["date_order"]].agg("max")
        now = max(recency["date_order"])
        recency_days = now - recency["date_order"]
        recency = pd.DataFrame(recency_days)
        recency.rename(columns={"date_order": "recency"}, inplace=True)
        recency["recency"] = recency["recency"].dt.days
        return recency

    def add_frequency(self) -> pd.DataFrame:
        frequency = self.df.groupby(["client_id"])[["date_order"]].count()
        frequency.rename(columns={"date_order": "frequency"}, inplace=True)
        return frequency

    def add_monetary(self) -> pd.DataFrame:
        monetary = self.df[["client_id", "sales_net"]]
        monetary = monetary.groupby(["client_id"]).sum()
        monetary.rename(columns={"sales_net": "monetary"}, inplace=True)
        return monetary

    def add_rfm(self) -> pd.DataFrame:
        print("adding recency...")
        recency = self.add_recency()
        print("adding frequency...")
        frequency = self.add_frequency()
        print("adding monetary...")
        monetary = self.add_monetary()
        rfm = pd.concat([recency, frequency, monetary], axis=1, ignore_index=False)
        # scaler = MinMaxScaler()
        # rfm = pd.DataFrame(scaler.fit_transform(rfm))
        rfm.index = recency.index
        rfm.columns = ["recency", "frequency", "monetary"]
        rfm["r_score"] = pd.qcut(
            rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
        )
        rfm["f_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        )
        rfm["m_score"] = pd.qcut(
            rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        )
        rfm["r_score"] = rfm["r_score"].astype(int)
        rfm["f_score"] = rfm["f_score"].astype(int)
        rfm["m_score"] = rfm["m_score"].astype(int)
        self.rfm = rfm
        rfm["cluster"] = "need attention"
        rfm.loc[
            (rfm["f_score"] > 3) & (rfm["r_score"] > 3) & (rfm["m_score"] > 3),
            "cluster",
        ] = "champions"
        rfm.loc[
            (rfm["f_score"] < 2.5) & (rfm["r_score"] < 2.5) & (rfm["m_score"] < 2.5),
            "cluster",
        ] = "to be lost"
        rfm.loc[
            (rfm["f_score"] < 2) & (rfm["r_score"] > 4) & (rfm["m_score"] < 2),
            "cluster",
        ] = "new comers"
        return rfm

    def price_most_bought_product(self):
        most_bought_product_by_client = self.df.groupby(
            "client_id", as_index=False
        ).apply(lambda x: x.loc[x["quantity"].idxmax()])
        most_bought_product_by_client = most_bought_product_by_client[
            ["client_id", "sales_net"]
        ]
        most_bought_product_by_client = most_bought_product_by_client.rename(
            columns={"sales_net": "price_most_bought_product"}
        )
        return most_bought_product_by_client

    def total_sales_tot_by_client_branch(
        self,
    ):  # compute total sales per client and branch
        total_sales_tot_by_client_and_branch = self.df.groupby(
            ["client_id", "order_channel"], as_index=True
        )[["sales_tot"]].sum()
        total_sales_tot_by_client_and_branch = (
            total_sales_tot_by_client_and_branch.unstack(level=-1)
        )
        new_cols = [
            "_".join(map(str, col))
            for col in total_sales_tot_by_client_and_branch.columns
        ]
        total_sales_tot_by_client_and_branch.columns = new_cols
        total_sales_tot_by_client_and_branch = (
            total_sales_tot_by_client_and_branch.fillna(0).reset_index()
        )
        return total_sales_tot_by_client_and_branch

    def n_unique_products_client(self):
        # Compute the number of unique products bought for each client
        number_of_products_bought_by_client = self.df.groupby(
            "client_id", as_index=False
        )[["product_id"]].nunique()
        number_of_products_bought_by_client.rename(
            columns={"product_id": "n_unique_products_bought"}, inplace=True
        )
        return number_of_products_bought_by_client

    def identify_churn(self):
        # Identify the clients whose maximum order date is more than 6 months ago
        max_order_date_by_client = self.df.groupby("client_id")[["date_order"]].max()
        cutoff_date = max(max_order_date_by_client["date_order"]) - pd.Timedelta(
            days=180
        )
        max_order_date_by_client["churn"] = 0
        max_order_date_by_client.loc[
            max_order_date_by_client.date_order <= cutoff_date, "churn"
        ] = 1
        churn = max_order_date_by_client[["churn"]]
        return churn

    def full_preprocessing(self):
        print("cleaning dates, adding year and month...")
        self.clean_dates()
        print("adding total sales...")
        self.add_sales_tot()
        print("dropping null rows...")
        self.drop_null_rows()
        print("removing negative rows...")
        self.remove_negative_rows()
        print("calculating rfm...")
        rfm = self.add_rfm()
        print("calculating price of most bought product...")
        most_bought_product_by_client = self.price_most_bought_product()
        print("calculating total_sales_tot_by_client_branch...")
        total_sales_tot_by_client_branch = self.total_sales_tot_by_client_branch()
        print("calculating the number of unique products bought for each client...")
        n_unique_products_client = self.n_unique_products_client()
        print("identifying churn...")
        churn = self.identify_churn()
        print("merging everything...")
        output = pd.merge(
            most_bought_product_by_client,
            total_sales_tot_by_client_branch,
            on="client_id",
        )
        output = pd.merge(output, n_unique_products_client, on="client_id")
        output = pd.merge(output, rfm, on="client_id")
        output = pd.merge(output, churn, on="client_id")
        self.preprocessed_df = output
        return output

    def clustering(self, n_clusters, columns):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(self.rfm.loc[:, columns])
        self.model = kmeans
        return kmeans

    def draw_clusters(self):
        self.rfm["cluster"] = self.model.labels_
        melted_rfm = pd.melt(
            self.rfm.reset_index(),
            id_vars=["client_id", "cluster"],
            value_vars=["r_score", "f_score", "m_score"],
            var_name="Features",
            value_name="Value",
        )
        sns.lineplot(melted_rfm, x="Features", y="Value", hue="cluster")
        plt.legend()
        self.clusters = melted_rfm
        return melted_rfm
