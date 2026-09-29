@echo off
title Visao Cidada YOLO - Sistema Profissional
chcp 65001 > nul
echo =======================================================
echo         VISAO CIDADA YOLO - INICIALIZADOR
echo =======================================================
echo.
echo Iniciando a interface web profissional...
echo O navegador abrira automaticamente.
echo Para encerrar o sistema, feche esta janela.
echo.
cd /d "%~dp0visao_cidada_yolo"
python -m streamlit run app.py
pause
