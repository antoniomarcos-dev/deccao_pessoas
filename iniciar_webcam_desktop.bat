@echo off
title Visao Cidada YOLO - Webcam Desktop
chcp 65001 > nul
echo =======================================================
echo     VISAO CIDADA YOLO - WEBCAM DESKTOP COM HUD
echo =======================================================
echo.
echo Controles da Janela:
echo [S] Salvar Foto/Snapshot   ^| [B] Alternar Privacidade Facial
echo [C] Resetar Pico de Pessoas ^| [ESPACO] Congelar/Pausar
echo [Q] ou [ESC] Encerrar
echo.
cd /d "%~dp0visao_cidada_yolo"
python webcam.py
pause
