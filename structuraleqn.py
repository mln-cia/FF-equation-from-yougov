import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import statsmodels.api as sm
import requests
import pyreadstat
import tempfile
import json
from sklearn.preprocessing import MinMaxScaler

@st.cache_data()
def authenticate(user, pwd):
    url = "https://api.brandindex.com/v1/auth/login"
    headers = {"accept": "application/json",
               "Content-Type": "application/json"}
    data = {
        "data": {"email": user,
                 "password": pwd},
        "meta": {"version": "v1"},
    }
    with requests.Session() as session:
        auth_response = session.post(url, json=data, headers=headers)
        if auth_response.status_code == 200:
            st.success("Authentication Successful!")
            return session
        else:
            st.error("Authentication Failed")
        return session


@st.cache_data()
def get_region(_session):
    regions_data = []
    regions = session.get(
        url="https://api.brandindex.com/v1/taxonomies/regions?all=true",
        headers={"accept": "application/json",
                 "Content-Type": "application/json"},
    ).json()["data"]

    for key_region, value_region in regions.items():
        if value_region["permitted"]:
            regions_data.append([value_region["label"], value_region["name"]])

    regions_data = dict(regions_data)

    return regions_data


@st.cache_data()
def get_sectors(_session, selected_region):
    sectors_data = []
    url_sectors = (
        f"https://api.brandindex.com/v1/taxonomies/regions/{selected_region}" +
        "/sectors"
    )

    sectors = session.get(url=url_sectors, headers=headers).json()["data"]

    for key_sector, value_sector in sectors.items():
        if value_sector["is_active"]:
            sectors_data.append([value_sector["label"], value_sector["id"]])

    sectors_data = dict(sectors_data)

    return sectors_data


@st.cache_data()
def get_brands(_session, selected_region, selected_sector):
    brands_data = []
    url_brands = f"https://api.brandindex.com/v1/taxonomies/regions/{selected_region}/sectors/{selected_sector}/brands"

    brands = session.get(url=url_brands, headers=headers).json()["data"]

    for key_brand, value_brand in brands.items():
        if value_brand["is_active"]:
            brands_data.append([value_brand["label"], value_brand["id"]])

    brands_data = dict(brands_data)

    return brands_data


@st.cache_data()
def retrieve_data(brands_list, start_date, end_date):
    my_bar = st.progress(0, text="Retrieving Data")
    for n, entry in enumerate(brands_list):
        metadata["data"]["queries"][0]["entity"]["brand_id"] = brands_data[entry]
        metadata["data"]["queries"][1]["entity"]["brand_id"] = brands_data[entry]

        metadata["data"]["queries"][0]["period"]["start_date"][
            "date"
        ] = start_date.strftime("%Y-%m-%d")
        metadata["data"]["queries"][1]["period"]["start_date"][
            "date"
        ] = start_date.strftime("%Y-%m-%d")

        metadata["data"]["queries"][0]["period"]["end_date"][
            "date"
        ] = end_date.strftime("%Y-%m-%d")
        metadata["data"]["queries"][1]["period"]["end_date"][
            "date"
        ] = end_date.strftime("%Y-%m-%d")

        try:
            response = session.post(
                url="https://api.brandindex.com/v1/analyses/execute",
                headers={
                    "accept": "application/octet-stream",
                    "Content-Type": "application/json",
                },
                json=metadata,
            )
            my_bar.progress(
                int(100 / len(brands_list) * (n)),
                text=f"Retrieving {entry} Data"
            )

            # st.success('Data for {} retrieved successfully'.format(entry))
        except Exception:
            st.error("Error encounter for {}".format(entry))

        index = response.json()["data"]["queries"][0]["data"]["coordinates"]["date"]
        columns = response.json()["data"]["queries"][0]["data"]["coordinates"]["metric"]
        values = np.array(response.json()["data"]["queries"][0]["data"]["values"])[
            :, :, 2
        ].T

        df = pd.DataFrame(values, columns=columns, index=index)
        df.index = pd.to_datetime(df.index)
        # df['Brand'] = entry

        df["Familiarity"] = np.array(
            response.json()["data"]["queries"][1]["data"]["values"]
        )[:, :, 2].T
        df["Feeling"] = df["buzz"]
        df["Favorability"] = df["consider"]
        df["Fervor"] = (df["buzz"] + df["wom"]) / 2
        df["Facilitation"] = (df["satisfaction"] + df["likelybuy"]) / 2
        df["Fascination"] = df["adaware"]
        df["Following"] = df["recommend"]

        df["CurrentCustomer"] = df["current_own"]

        df = (
            df[
                [
                    "Familiarity",
                    "Feeling",
                    "Favorability",
                    "Fervor",
                    "Facilitation",
                    "Fascination",
                    "Following",
                    "CurrentCustomer",
                ]
            ]
            .rolling(window=7)
            .mean()
            .dropna()
        )
        df["Brand"] = entry
        # st.write(df)
        # df = df.set_index('Brand')

        dfs.append(df)

    my_bar.progress(100, text="Data Retrieved")

    final_df = pd.concat(dfs)

    st.write(final_df)

    return final_df


# Selettore per scegliere tra survey o dati YouGov nella barra laterale
st.title("Structural Equation Creator")
# st.sidebar.title("Seleziona il tipo di dati per costruire un'equazione")
# scelta = st.sidebar.selectbox("Seleziona la fonte dei dati",
#                               ["Survey", "Dati YouGov"])



with st.sidebar.expander(
    "Reminder on how F&F metrics are calculated from YouGov", expanded=False
):
    st.write(
        """
    | Dimensione                | Metriche YG                              |
    |---------------------------|------------------------------------------|
    | Familiarity               | Awareness                                |
    | Feeling                   | Attention                                |
    | Favourability             | Consideration                            |
    | Fervor                    | Media fra WOM Exposure e Buzz            |
    | Facilitation              | Media fra Satisfaction e Purchase Intent |
    | Fascination               | Ad Awareness                             |
    | Following                 | Recommend                                |
    """
    )

############################################################################
with st.sidebar:
    user = st.text_input('User')
    pwd = st.text_input('Password', type='password')
if user and pwd:
    session = authenticate(user, pwd)

    headers = {"accept": "application/json"}

    metadata_path = (
        "./metadata.json"
    )

    # Load the JSON file
    with open(metadata_path, "r") as json_file:
        metadata = json.load(json_file)

    ###########################
    col1, col2 = st.columns(2)
    with col1:
        regions_data = get_region(session)
        selected_region = st.selectbox(
            label="Select Region", options=regions_data.keys(), index=None
        )

        if selected_region is not None:
            sectors_data = get_sectors(session, regions_data[selected_region])
            selected_sector = st.selectbox(
                label="Select Sector", options=sectors_data.keys(), index=None
            )

            if selected_sector is not None:
                brands_data = get_brands(
                    session,
                    regions_data[selected_region],
                    sectors_data[selected_sector],
                )
                selected_brands = st.multiselect(
                    label="Select Brands", options=brands_data.keys()
                )

                metadata["data"]["queries"][0]["entity"]["sector_id"] = sectors_data[
                    selected_sector
                ]
                metadata["data"]["queries"][1]["entity"]["sector_id"] = sectors_data[
                    selected_sector
                ]

                metadata["data"]["queries"][0]["entity"]["region"] = regions_data[
                    selected_region
                ]
                metadata["data"]["queries"][1]["entity"]["region"] = regions_data[
                    selected_region
                ]

    with col2:
        import datetime

        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime(datetime.datetime.today()) + pd.DateOffset(years=-1),
        )
        end_date = st.date_input(
            "End Date", value=pd.to_datetime(pd.to_datetime(datetime.datetime.today()))
        )

    dfs = []

    go = st.checkbox("Confirm Brands")

    if go:
        final_df = retrieve_data(selected_brands, start_date, end_date)

        scaler = MinMaxScaler().set_output(transform="pandas")
        final_df.iloc[:, :-1] = scaler.fit_transform(final_df.iloc[:, :-1]) * 100
        # final_df = final_df.rolling(window=14).mean()

                # Creazione dello scatterplot
        fig, axes = plt.subplots(4, 2, figsize=(15, 15))
        axes = axes.flatten()  # Rende l'array bidimensionale piatto
        i = 0
        coefficients = []

        for f in final_df.columns[:-2]:
            X = final_df[[f]]
            y = final_df["CurrentCustomer"]

            # Aggiungi un'intercetta per la regressione
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            predictions = model.predict(X)

            ax = axes[i]
            sns.scatterplot(
                data=final_df,
                x=f,
                y="CurrentCustomer",
                alpha=0.6,
                ax=ax,
                hue="Brand",
            )

            # Traccia la linea di regressione
            ax.plot(final_df[f], predictions, color="red",
                    label="Linear Fit")

            # Imposta i limiti di x solo se il limite inferiore è sotto 0
            current_xlim = ax.get_xlim()
            if current_xlim[0] < 0:
                ax.set_xlim(left=0)

            # Imposta i limiti di y (sempre inferiori a 0)
            ax.set_ylim(bottom=0)

            ax.set_title(f"{f} vs Current Customer")
            ax.set_xlabel(f)
            ax.set_ylabel("Current Customer")
            ax.legend()

            # Salva il coefficiente (senza intercetta) per ogni variabile F
            coefficients.append(
                {
                    "Dimensione": f,
                    "Coefficiente": model.params[1],
                    "R^2": model.rsquared,
                }
            )
            i += 1
        plt.tight_layout()
        st.pyplot(fig)

        st.write('Per valutare la Findability è necessario caricare il file SPSS contenente la domanda relativa.'
                 'Selezionare la domanda relativa lla variabile Findability e alla variabile di Current Customer.')
        # Carica il file
        dataset = st.file_uploader("Upload File", type=["sav"])

        # Verifica se il file è stato caricato
        if dataset is not None:
            # Crea un file temporaneo
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                # Scrivi il contenuto dell'UploadedFile nel file temporaneo
                tmp_file.write(dataset.getvalue())
                # Ottieni il percorso del file temporaneo
                tmp_file_path = tmp_file.name

            # Leggi il file temporaneo con pyreadstat
            df, meta = pyreadstat.read_sav(tmp_file_path)
            df.index = df["record"]

            options = set(
                i.split(":")[0].split("r")[0] + i.split("-")[-1].split(":")[-1]
                for i in meta.column_names_to_labels.values()
            )
            dict_options = {
                i.split(":")[0].split("r")[0] + i.split("-")[-1].split(":")[-1]: i
                for i in meta.column_names_to_labels.values()
            }

            questions_findability = st.selectbox(
                label="Seleziona la domanda relativa alla variabile Findability",
                options=options,
                index=None,
            )
            question_currentcustomer = st.selectbox(
                label="Seleziona la domanda relativa alla variabile " +
                "Current Customer",
                options=options,
                index=None,
            )

            dict_rename_cols = {
                key: val.split(":")[-1].split("-")[0].strip()
                for key, val in meta.column_names_to_labels.items()
            }
            df_findability = df[
                [
                    col
                    for col in df.columns
                    if questions_findability.split(" ", 1)[0] + "r" in col
                ]
            ].rename(columns=dict_rename_cols)

            df_currentcustomer = df[
                [
                    col
                    for col in df.columns
                    if question_currentcustomer.split(" ", 1)[0] + "r" in col
                ]
            ].rename(columns=dict_rename_cols)

            final_df = pd.concat(
                 [df_findability.sum()/df_findability.count(), df_currentcustomer.sum()/df_currentcustomer.count()], axis=1
            )
            final_df.columns = ['Findability', 'CurrentCustomer']

            X_fit = final_df[["Findability"]]
            y = final_df["CurrentCustomer"]

            X_fit = sm.add_constant(X_fit)
            model = sm.OLS(y, X_fit).fit()
            predictions = model.predict(X_fit)

            coefficients.append(
                {
                    "Dimensione": 'Findability',
                    "Coefficiente": model.params[1],
                    "R^2": model.rsquared,
                }
            )

            # plt.tight_layout()
            # st.pyplot(fig)

            st.write('## Final Table')
            # Crea una tabella con i coefficienti di regressione
            coef_df = pd.DataFrame(coefficients)
            coef_df["F %Importance"] = (
                coef_df.Coefficiente / coef_df.Coefficiente.sum() * 100
            )
            st.write(coef_df)


