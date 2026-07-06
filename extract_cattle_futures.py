import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


DATA_DIR = "data"

MARKETS = {
    "feeder_cattle": {
        "ticker": "GF=F",
        "output_file": "feeder_cattle.csv",
        "market_name": "Feeder Cattle Futures",
    },
    "live_cattle": {
        "ticker": "LE=F",
        "output_file": "live_cattle.csv",
        "market_name": "Live Cattle Futures",
    },
}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_history(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Descarga histórico diario de los últimos años usando yfinance.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 10)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No se descargaron datos para {ticker}")

    df = df.reset_index()

    # Algunas versiones de yfinance pueden regresar columnas tipo MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]

    return df


def clean_history(df: pd.DataFrame, ticker: str, market_name: str) -> pd.DataFrame:
    """
    Limpia la base y agrega columnas de año, mes y año-mes.
    """
    df = df.copy()

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    df = df.rename(columns=rename_map)

    expected_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    existing_cols = [col for col in expected_cols if col in df.columns]
    df = df[existing_cols]

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Quitar filas sin precio
    df = df.dropna(subset=["close"])

    # Columnas para dashboard
    df["ticker"] = ticker
    df["market"] = market_name
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["month_name"] = pd.to_datetime(df["date"]).dt.strftime("%B")
    df["year_month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")

    final_cols = [
        "date",
        "ticker",
        "market",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "year",
        "month",
        "month_name",
        "year_month",
    ]

    df = df[[col for col in final_cols if col in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def merge_with_existing(new_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Une con archivo existente, elimina fechas duplicadas y conserva la más reciente.
    """
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        old_df["date"] = pd.to_datetime(old_df["date"]).dt.date

        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date

        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)

        return combined

    return new_df


def save_market_data(config: dict):
    ticker = config["ticker"]
    output_file = config["output_file"]
    market_name = config["market_name"]

    print(f"Descargando {market_name} ({ticker})...")

    raw_df = download_history(ticker=ticker, years=10)
    clean_df = clean_history(raw_df, ticker=ticker, market_name=market_name)

    output_path = os.path.join(DATA_DIR, output_file)

    final_df = merge_with_existing(clean_df, output_path)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Archivo actualizado: {output_path}")
    print(f"Filas totales: {len(final_df)}")
    print(f"Fecha inicial: {final_df['date'].min()}")
    print(f"Fecha final: {final_df['date'].max()}")
    print("-" * 50)


def main():
    ensure_data_dir()

    for config in MARKETS.values():
        save_market_data(config)


if __name__ == "__main__":
    main()
