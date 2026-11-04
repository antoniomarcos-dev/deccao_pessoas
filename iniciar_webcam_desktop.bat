@echo off
title Visao Cidada YOLO - Webcam Desktop
chcp 65001 > nul
echo =======================================================
echo     VISAO CIDADA YOLO - WEBCAM DESKTOP COM HUD
echo =======================================================
echo.
echo Controles da Janela em Tempo Real:
echo [S] Salvar Foto/Snapshot    ^| [B] Privacidade Facial (LGPD)
echo [D] Modo Alta Densidade     ^| [T] Rastreamento de Pessoas Unicas
echo [H] Mapa de Calor Termico   ^| [L] Realce Noturno (Low-Light)
echo [+] / [-] Ajustar Confianca ^| [C] Resetar Pico e Contadores
echo [ESPACO] Congelar/Pausar    ^| [Q] ou [ESC] Encerrar
echo.
cd /d "%~dp0visao_cidada_yolo"
python webcam.py
pause
