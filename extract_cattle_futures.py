import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


DATA_DIR = "data"

YEARS_BACK = 10

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
    """
    Crea la carpeta data si no existe.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def download_history(ticker: str, years: int = YEARS_BACK) -> pd.DataFrame:
    """
    Descarga histórico diario de Yahoo Finance usando yfinance.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 10)

    print(f"Descargando {ticker} desde {start_date.date()} hasta {end_date.date()}...")

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

    # Algunas versiones de yfinance regresan columnas MultiIndex.
    # Esto las aplana para evitar errores.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[0] else col[1]
            for col in df.columns
        ]

    return df


def clean_history(df: pd.DataFrame, ticker: str, market_name: str) -> pd.DataFrame:
    """
    Limpia la base y agrega columnas útiles para dashboard.
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

    if "date" not in df.columns:
        raise ValueError(f"No se encontró columna de fecha para {ticker}")

    if "close" not in df.columns:
        raise ValueError(f"No se encontró columna close para {ticker}")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Quitar filas sin precio de cierre.
    df = df.dropna(subset=["close"])

    # Columnas para identificar mercado.
    df["ticker"] = ticker
    df["market"] = market_name

    # Columnas para visualización.
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["month_name"] = pd.to_datetime(df["date"]).dt.strftime("%B")
    df["year_month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")

    # Orden final.
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
    Si ya existe un CSV previo, lo une con la nueva descarga,
    elimina fechas duplicadas y conserva la versión más reciente.
    """
    if not os.path.exists(output_path):
        return new_df

    old_df = pd.read_csv(output_path)

    if old_df.empty:
        return new_df

    old_df["date"] = pd.to_datetime(old_df["date"]).dt.date
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

    combined = pd.concat([old_df, new_df], ignore_index=True)

    # Como cada archivo es de un solo mercado, date basta para quitar duplicados.
    combined = combined.drop_duplicates(subset=["date"], keep="last")

    combined = combined.sort_values("date").reset_index(drop=True)

    return combined


def save_market_data(config: dict) -> pd.DataFrame:
    """
    Descarga, limpia y guarda un mercado individual.
    """
    ticker = config["ticker"]
    output_file = config["output_file"]
    market_name = config["market_name"]

    print("=" * 60)
    print(f"Procesando {market_name} ({ticker})")

    raw_df = download_history(ticker=ticker, years=YEARS_BACK)
    clean_df = clean_history(raw_df, ticker=ticker, market_name=market_name)

    output_path = os.path.join(DATA_DIR, output_file)

    final_df = merge_with_existing(clean_df, output_path)

    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Archivo actualizado: {output_path}")
    print(f"Filas totales: {len(final_df)}")
    print(f"Fecha inicial: {final_df['date'].min()}")
    print(f"Fecha final: {final_df['date'].max()}")

    return final_df


def create_combined_file():
    """
    Crea un CSV combinado con Feeder Cattle y Live Cattle.
    Este archivo es el recomendado para Looker Studio.
    """
    print("=" * 60)
    print("Creando archivo combinado...")

    dfs = []

    for config in MARKETS.values():
        path = os.path.join(DATA_DIR, config["output_file"])

        if not os.path.exists(path):
            print(f"No existe todavía: {path}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"Archivo vacío: {path}")
            continue

        dfs.append(df)

    if not dfs:
        print("No hay archivos disponibles para combinar.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    # En el combinado sí quitamos duplicados usando date + ticker,
    # porque hay dos mercados distintos.
    combined = combined.drop_duplicates(
        subset=["date", "ticker"],
        keep="last"
    )

    combined = combined.sort_values(["market", "date"]).reset_index(drop=True)

    output_path = os.path.join(DATA_DIR, "cattle_futures_combined.csv")

    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Archivo combinado actualizado: {output_path}")
    print(f"Filas combinadas: {len(combined)}")
    print(f"Fecha inicial: {combined['date'].min()}")
    print(f"Fecha final: {combined['date'].max()}")


def main():
    ensure_data_dir()

    for config in MARKETS.values():
        save_market_data(config)

    create_combined_file()

    print("=" * 60)
    print("Proceso terminado correctamente.")


if __name__ == "__main__":
    main()
