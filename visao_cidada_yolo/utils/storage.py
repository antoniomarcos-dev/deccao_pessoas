"""
Camada de Persistência e Telemetria (SQLite + Sincronização CSV Thread-Safe).
Inclui desduplicação inteligente contra re-execuções de interface (Streamlit reruns),
migração automática de dados legados e consultas analíticas de alta velocidade.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

import pandas as pd

from config import DEVICE, DEFAULT_MODEL_NAME, LOGS_DIR, RECORDS_FILE, ensure_project_directories

DB_FILE = LOGS_DIR / "visao_cidada.db"

EXPECTED_COLUMNS = [
    "id",
    "data",
    "horario",
    "origem",
    "tipo",
    "quantidade",
    "pessoas_unicas",
    "cenario",
    "confianca_minima",
    "confianca_media",
    "latencia_ms",
    "dispositivo",
    "modelo",
]

LEGACY_CSV_COLUMNS = [
    "data",
    "horario",
    "origem",
    "tipo",
    "quantidade",
    "confianca_minima",
    "dispositivo",
]


def classify_scenario_label(count: int) -> str:
    """Retorna o rótulo textual padronizado de cenário conforme a ocupação."""
    if count <= 0:
        return "Ambiente Livre"
    if count == 1:
        return "Individual"
    if count <= 4:
        return "Em Grupo"
    if count <= 12:
        return "Grupo Grande"
    return "Aglomeração"


class DetectionRepository:
    """Gerenciador thread-safe de registros de detecção em SQLite com espelhamento CSV."""

    def __init__(self, db_path: Path = DB_FILE, csv_path: Path = RECORDS_FILE) -> None:
        self.db_path = db_path
        self.csv_path = csv_path
        self._lock = threading.RLock()
        self._last_signature: tuple[Any, ...] | None = None
        self._last_timestamp: float = 0.0
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        ensure_project_directories()
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data TEXT NOT NULL,
                        horario TEXT NOT NULL,
                        origem TEXT NOT NULL,
                        tipo TEXT NOT NULL,
                        quantidade INTEGER NOT NULL,
                        pessoas_unicas INTEGER NOT NULL DEFAULT 0,
                        cenario TEXT NOT NULL DEFAULT 'Ambiente Livre',
                        confianca_minima REAL NOT NULL DEFAULT 0.3,
                        confianca_media REAL NOT NULL DEFAULT 0.0,
                        latencia_ms REAL NOT NULL DEFAULT 0.0,
                        dispositivo TEXT NOT NULL DEFAULT 'cpu',
                        modelo TEXT NOT NULL DEFAULT 'best.pt'
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_data ON detections(data);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_tipo ON detections(tipo);")

                # Se o banco estiver vazio mas existir CSV legado, importa sem duplicatas consecutivas
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM detections")
                row_count = cursor.fetchone()["cnt"]
                if row_count == 0 and self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                    self._migrate_from_csv(conn)
            self._initialized = True

    def _migrate_from_csv(self, conn: sqlite3.Connection) -> None:
        """Importa registros do CSV existente, removendo duplicatas espúrias de reruns no mesmo segundo/minuto."""
        try:
            df = pd.read_csv(self.csv_path, on_bad_lines="skip")
            if df.empty:
                return

            prev_sig = None
            rows_to_insert = []
            for _, row in df.iterrows():
                data = str(row.get("data", datetime.now().date().isoformat()))
                horario = str(row.get("horario", "00:00:00"))
                origem = str(row.get("origem", "desconhecido"))
                tipo = str(row.get("tipo", "imagem"))
                try:
                    quantidade = int(float(row.get("quantidade", 0)))
                except (ValueError, TypeError):
                    quantidade = 0
                try:
                    conf_min = round(float(row.get("confianca_minima", 0.3)), 2)
                except (ValueError, TypeError):
                    conf_min = 0.3
                dispositivo = str(row.get("dispositivo", "cpu"))
                pessoas_unicas = int(row.get("pessoas_unicas", quantidade)) if "pessoas_unicas" in row and pd.notna(row.get("pessoas_unicas")) else quantidade
                cenario = str(row.get("cenario", classify_scenario_label(quantidade)))
                conf_media = float(row.get("confianca_media", 0.0)) if "confianca_media" in row and pd.notna(row.get("confianca_media")) else 0.0
                latencia = float(row.get("latencia_ms", 0.0)) if "latencia_ms" in row and pd.notna(row.get("latencia_ms")) else 0.0
                modelo = str(row.get("modelo", DEFAULT_MODEL_NAME)) if "modelo" in row and pd.notna(row.get("modelo")) else DEFAULT_MODEL_NAME

                # Filtra duplicatas idênticas consecutivas geradas por reruns do Streamlit
                sig = (data, origem, tipo, quantidade, conf_min, dispositivo)
                if sig == prev_sig:
                    continue
                prev_sig = sig

                rows_to_insert.append((
                    data,
                    horario,
                    origem,
                    tipo,
                    quantidade,
                    pessoas_unicas,
                    cenario,
                    conf_min,
                    conf_media,
                    latencia,
                    dispositivo,
                    modelo,
                ))

            if rows_to_insert:
                conn.executemany(
                    """
                    INSERT INTO detections (
                        data, horario, origem, tipo, quantidade, pessoas_unicas,
                        cenario, confianca_minima, confianca_media, latencia_ms, dispositivo, modelo
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_insert,
                )
        except Exception:
            pass

    def register(
        self,
        source: str | Path,
        input_type: str,
        count: int,
        confidence: float,
        device: str = DEVICE,
        unique_people: int | None = None,
        avg_confidence: float = 0.0,
        latency_ms: float = 0.0,
        model_name: str = DEFAULT_MODEL_NAME,
        cooldown_seconds: float = 8.0,
    ) -> bool:
        """
        Registra uma detecção no banco SQLite e no arquivo CSV.
        Ignora chamadas duplicadas idênticas dentro da janela de `cooldown_seconds`.
        Retorna True se o registro foi gravado, False se foi filtrado por desduplicação.
        """
        self.initialize()
        source_clean = Path(str(source)).name or str(source)
        count_int = max(0, int(count))
        unique_int = max(count_int, int(unique_people)) if unique_people is not None else count_int
        conf_round = round(float(confidence), 2)
        avg_conf_round = round(float(avg_confidence), 4)
        lat_round = round(float(latency_ms), 1)
        scenario = classify_scenario_label(count_int)
        model_clean = Path(str(model_name)).name or str(model_name)

        signature = (source_clean, input_type, count_int, conf_round, str(device), model_clean)
        now_perf = time.perf_counter()

        with self._lock:
            if (
                cooldown_seconds > 0
                and self._last_signature == signature
                and (now_perf - self._last_timestamp) < cooldown_seconds
            ):
                return False

            self._last_signature = signature
            self._last_timestamp = now_perf

            now_dt = datetime.now()
            date_str = now_dt.date().isoformat()
            time_str = now_dt.strftime("%H:%M:%S")

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO detections (
                        data, horario, origem, tipo, quantidade, pessoas_unicas,
                        cenario, confianca_minima, confianca_media, latencia_ms, dispositivo, modelo
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_str,
                        time_str,
                        source_clean,
                        input_type,
                        count_int,
                        unique_int,
                        scenario,
                        conf_round,
                        avg_conf_round,
                        lat_round,
                        str(device),
                        model_clean,
                    ),
                )

            self._append_to_csv(
                date_str=date_str,
                time_str=time_str,
                source_clean=source_clean,
                input_type=input_type,
                count_int=count_int,
                unique_int=unique_int,
                scenario=scenario,
                conf_round=conf_round,
                avg_conf_round=avg_conf_round,
                lat_round=lat_round,
                device=str(device),
                model_clean=model_clean,
            )
            return True

    def _append_to_csv(
        self,
        date_str: str,
        time_str: str,
        source_clean: str,
        input_type: str,
        count_int: int,
        unique_int: int,
        scenario: str,
        conf_round: float,
        avg_conf_round: float,
        lat_round: float,
        device: str,
        model_clean: str,
    ) -> None:
        """Mantém `logs/detections.csv` atualizado e sincronizado."""
        try:
            ensure_project_directories()
            new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            # Verifica se o cabeçalho existente possui as novas colunas; caso contrário, sincroniza completo
            if not new_file:
                with self.csv_path.open("r", encoding="utf-8", errors="ignore") as f:
                    header_line = f.readline().strip()
                if "cenario" not in header_line:
                    self._export_full_csv()
                    return

            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if new_file:
                    writer.writerow([col for col in EXPECTED_COLUMNS if col != "id"])
                writer.writerow([
                    date_str,
                    time_str,
                    source_clean,
                    input_type,
                    count_int,
                    unique_int,
                    scenario,
                    conf_round,
                    avg_conf_round,
                    lat_round,
                    device,
                    model_clean,
                ])
        except Exception:
            pass

    def _export_full_csv(self) -> None:
        """Reescreve o CSV completo a partir do banco SQLite com o schema atualizado."""
        try:
            df = self.load_dataframe()
            if df.empty:
                with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow([col for col in EXPECTED_COLUMNS if col != "id"])
            else:
                cols = [c for c in EXPECTED_COLUMNS if c != "id" and c in df.columns]
                df[cols].to_csv(self.csv_path, index=False, encoding="utf-8")
        except Exception:
            pass

    def load_dataframe(self, limit: int = 5000) -> pd.DataFrame:
        """Carrega os registros do banco em um DataFrame Pandas padronizado."""
        self.initialize()
        with self._lock:
            try:
                with self._get_connection() as conn:
                    df = pd.read_sql_query(
                        f"SELECT * FROM detections ORDER BY id ASC LIMIT {int(limit)}",
                        conn,
                    )
                if df.empty:
                    return pd.DataFrame(columns=EXPECTED_COLUMNS)
                for col in EXPECTED_COLUMNS:
                    if col not in df.columns:
                        df[col] = "N/A"
                return df
            except Exception:
                return pd.DataFrame(columns=EXPECTED_COLUMNS)

    def clear_all(self) -> None:
        """Remove todos os registros do histórico (SQLite e CSV)."""
        self.initialize()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM detections;")
            self._last_signature = None
            self._export_full_csv()


_REPOSITORY_INSTANCE: DetectionRepository | None = None


def get_repository() -> DetectionRepository:
    global _REPOSITORY_INSTANCE
    if _REPOSITORY_INSTANCE is None:
        _REPOSITORY_INSTANCE = DetectionRepository()
    return _REPOSITORY_INSTANCE

