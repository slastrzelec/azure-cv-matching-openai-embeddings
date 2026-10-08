#!/bin/bash
pip install -r requirements.txt
python -m streamlit run app.py --server.port "${PORT:-8000}" --server.address 0.0.0.0 --server.headless true
