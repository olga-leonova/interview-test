import pandas as pd


def calculate_save_price_rank_info(df: pd.DataFrame, path: str) -> None:
    """
    Rank based on column “Price”, grouped by column “brand”,
    update dataset and save it as csv file using giving path
    :param df: input dataset
    :param path: path where updated dataset should be stored
    :return:
    """
    df['price_rank'] = df.groupby('brand')['price'].rank()

    df.to_csv(f'{path}/price_rank_info.csv', index=False)


def calculate_save_min_max_hdd_info(df: pd.DataFrame, path: str) -> None:
    """
    Calculate minimum and maximum of column “HDD_GB” and save as csv file it using giving path
    :param df: input dataset
    :param path: path where updated dataset should be stored
    :return:
    """
    value_list = [df['hdd_gb'].min(), df['hdd_gb'].max()]
    df_tmp = pd.DataFrame([value_list],
                          columns=['hdd_gb_min', 'hdd_gb_max'])

    df_tmp.to_csv(f'{path}/min_max_hdd_info.csv', index=False)


def calculate_save_median_ghz_by_ram_info(df: pd.DataFrame, path: str) -> None:
    """
    Calculate median of column “GHz”, grouped by column “RAM_GB” and save it as csv file using giving path
    :param df: input dataset
    :param path: path where updated dataset should be stored
    :return:
    """
    df_tmp = df.groupby('ram_gb', as_index=False).agg({'ghz': 'median'})
    df_tmp.to_csv(f'{path}/median_ghz_by_ram_info.csv', index=False)
