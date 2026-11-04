"""
Gerenciador Assíncrono de Captura de Câmera / Stream em Thread Dedicada.
Elimina o atraso de buffer (lag) do OpenCV no Windows (DirectShow/MSMF) e fluxos RTSP,
garantindo que a IA sempre processe o frame mais recente disponível.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np


class ThreadedCamera:
    """
    Leitor de câmera não-bloqueante em thread de segundo plano (daemon).
    Mantém apenas o último frame capturado, descartando quadros antigos acumulados no buffer do driver.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps_target: int = 30,
    ) -> None:
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.height = height
        self.fps_target = fps_target

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ok: bool = False
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._frame_seq: int = 0

    def start(self) -> ThreadedCamera:
        """Abre o dispositivo de vídeo e inicia a thread de leitura contínua."""
        if self._running and self.isOpened():
            return self

        self.release()
        self._cap = self._open_capture(self.source)
        if self._cap is None or not self._cap.isOpened():
            self._ok = False
            self._running = False
            return self

        # Configura resolução e tamanho mínimo de buffer para baixa latência
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        except Exception:
            pass

        ok, first_frame = self._cap.read()
        with self._lock:
            self._ok = bool(ok and first_frame is not None)
            self._frame = first_frame.copy() if self._ok else None
            if self._ok:
                self._frame_seq = 1

        if self._ok:
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True, name=f"CamReader-{self.source}")
            self._thread.start()

        return self

    @staticmethod
    def _open_capture(source: int | str) -> cv2.VideoCapture | None:
        """Tenta abrir a câmera no Windows usando CAP_DSHOW primeiro (mais rápido e sem travamento), com fallback."""
        try:
            if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if cap.isOpened():
                    return cap
                cap.release()
                time.sleep(0.08)
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
        return None

    def _reader_loop(self) -> None:
        consecutive_failures = 0
        while self._running and self._cap is not None:
            try:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    consecutive_failures = 0
                    with self._lock:
                        self._ok = True
                        self._frame = frame
                        self._frame_seq += 1
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        with self._lock:
                            self._ok = False
                        break
                    time.sleep(0.01)
            except Exception:
                break
            # Pequeno yield para não saturar 1 núcleo de CPU se o driver responder instantaneamente
            time.sleep(0.002)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Retorna cópia segura do frame mais recente capturado."""
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def isOpened(self) -> bool:
        return bool(self._running and self._cap is not None and self._cap.isOpened())

    def set(self, prop_id: int, value: Any) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return bool(self._cap.set(prop_id, value))
        return False

    def get(self, prop_id: int) -> float:
        if self._cap is not None and self._cap.isOpened():
            return float(self._cap.get(prop_id))
        return 0.0

    def release(self) -> None:
        """Encerra a thread de captura e libera o driver da câmera com segurança."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        with self._lock:
            self._ok = False
            self._frame = None

