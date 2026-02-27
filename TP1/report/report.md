# Rapport de TP1 : Segmentation interactive avec SAM

**Lien du dépôt Git :** `[INSERER LE LIEN ICI]`
**Environnement d'exécution :** Nœud GPU via SLURM (Cluster TSP)

## 1. Initialisation
**Arborescence du projet :**
```
Environnement Conda : csc8608

CUDA : torch 2.1.0
cuda_available True
device_count 1

Import SAM : ok
sam_ok

Tunnel SSH et UI :
Port choisi : 8511
UI accessible via SSH tunnel : oui
Capture UI Streamlit : [INSERER CAPTURE STREAMLIT VIDE]

## 2. Dataset
Nombre d'images final : [X]
Sélection représentative :
image1.jpg : Cas simple avec un objet contrasté.
image2.jpg : Cas chargé (rue avec plusieurs éléments).

[INSERER 2 VIGNETTES: CAS SIMPLE ET DIFFICILE]

## 3. Chargement SAM
Modèle : vit_h
Checkpoint : sam_vit_h_4b8939.pth
Test rapide dans la console :[INSERER LOG DU SCRIPT QUICK_TEST_SAM.PY]
Commentaire : [Ajouter 3-5 lignes sur la vitesse et les limites observées]

## 4. Mesures et Visualisation
Test overlay console :[INSERER VIGNETTE DE L'OVERLAY]
ImageScoreAire (px
)Périmètreimg_01.jpg
0.9815000800.5
[REMPLIR LE TABLEAU]
Commentaire : [5-8 lignes : Dans quels cas l'overlay aide-t-il à débugger ?]

## 5. Mini-UI Streamlit
Tests Streamlit (captures et métriques) :
[INSERER 2-3 CAPTURES DE L'UI EN FONCTIONNEMENT]
ImageBBoxScoreAireTemps (ms)
[REMPLIR LE TABLEAU]
Débug BBox : [Qu'est-ce qui change quand vous modifiez la taille de la BBox ?]

## 6. Affinage (Points FG/BG)
Comparaison (Avant / Après) :
Image 1 (BBox seule) : [CAPTURE MASQUE ECHOUE]
Image 1 (BBox + Points) : [CAPTURE MASQUE REUSSI] -> Points utilisés : [(x,y,FG), (x,y,BG)]
Commentaire : [6-10 lignes : Quand les points BG sont-ils indispensables ? Quels cas échouent encore ?]

## 7. Bilan et réflexion (POC vers produit)
Analyse des échecs (3 facteurs) :
Actions concrètes pour améliorer : [8-12 lignes sur le post-traitement, dataset, UI]
Industrialisation (Monitoring & Logs) :
Si je devais passer cette brique en production, je monitorerais en priorité :
Temps d'inférence (Latency) : Pour détecter une surcharge GPU.
Score de confiance de SAM : Une baisse moyenne indique un drift sur la nature des images entrantes.
Nombre de masques modifiés/rejetés par l'utilisateur : Indique la pertinence réelle du modèle.
Volume des BBox (ratio bbox/image) : Pour repérer des usages anormaux de l'interface.
Erreurs OOM (Out Of Memory) GPU : Pour dimensionner correctement le matériel.