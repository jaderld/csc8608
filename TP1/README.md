# TP1 - Segmentation Interactive avec SAM

Ce dépôt contient le code source et le rapport pour le TP1 d'ingénierie de la vision par ordinateur, axé sur l'intégration du Segment Anything Model (SAM).



## Structure
- `data/images/` : Dataset de test
- `models/` : Checkpoints SAM (non versionnés)
- `src/` : Code source (utilitaires et app Streamlit)
- `outputs/overlays/` : Résultats sauvegardés
- `report/` : Rapport du TP

## Lancement rapide
1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer l'application : `streamlit run src/app.py --server.port 8511 --server.address 0.0.0.0`