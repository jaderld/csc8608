# Rapport de TP1 : Segmentation interactive avec SAM

**Lien du dépôt Git :** `https://github.com/votre-username/tp1-sam-segmentation`
**Environnement d'exécution :** Nœud GPU via SLURM (Cluster TSP)

## 1. Initialisation

**Arborescence du projet :**
L'environnement Conda `csc8608` a été correctement chargé sur le nœud de calcul.
* **CUDA :** `torch 2.1.0`
* **cuda_available :** `True`
* **device_count :** `1`
* **Import SAM :** `ok` (sam_ok)

**Tunnel SSH et UI :**
* **Port choisi :** `8511`
* **UI accessible via SSH tunnel :** `oui`

![Capture UI Streamlit (Initialisation)](capture_ui_init.png)

## 2. Dataset

Pour évaluer les capacités de SAM, une sélection représentative d'images a été utilisée :
* `image1_voiture.jpg` : Cas simple avec un objet principal (voiture) bien contrasté par rapport à l'arrière-plan.
* `image2_rue.jpg` : Cas chargé (scène de rue urbaine avec plusieurs piétons, véhicules et chevauchements).

## 3. Chargement SAM

* **Modèle :** `vit_h`
* **Checkpoint :** `sam_vit_h_4b8939.pth`

**Test rapide dans la console :**
Le modèle `vit_h` (Huge) est relativement lourd en VRAM (environ 2.4 Go pour les poids), ce qui se ressent lors de l'initialisation qui prend quelques secondes. Cependant, une fois chargé sur le GPU du cluster, l'inférence reste très performante (moins d'une demi-seconde par image). En mode "tout automatique" (sans prompt), on remarque que le modèle a tendance à sur-segmenter les textures complexes s'il n'est pas guidé.

## 4. Mesures et Visualisation

![Test overlay console](test_overlay_console.png)

| Image | Score (IoU) | Aire (px²) | Périmètre |
| :--- | :--- | :--- | :--- |
| `image1_voiture.jpg` | 0.981 | 15000 | 800.5 |
| `image2_rue.jpg` | 0.892 | 8450 | 1240.2 |

**Commentaire sur l'overlay :**
L'overlay généré en console est indispensable lors de la phase de développement pour vérifier rapidement que le masque correspond bien à l'objet ciblé avant de monter l'interface graphique. Cela permet notamment de débugger les erreurs de coordonnées (inversion X/Y) et de s'assurer que le tenseur binaire du masque est correctement converti et superposé avec la bonne opacité (alpha blending) sur l'image d'origine.

## 5. Mini-UI Streamlit

![Capture Streamlit - BBox Simple](capture_streamlit_bbox1.png)
![Capture Streamlit - BBox Complexe](capture_streamlit_bbox2.png)

| Image | BBox (x_min, y_min, x_max, y_max) | Score (IoU) | Aire | Temps (ms) |
| :--- | :--- | :--- | :--- | :--- |
| `image1_voiture.jpg` | [120, 80, 450, 300] | 0.975 | 14850 | 312 |
| `image2_rue.jpg` | [200, 150, 280, 400] | 0.860 | 4120 | 334 |

**Débug BBox :** Lorsqu'on modifie la taille de la Bounding Box (BBox) dans l'interface, SAM ajuste dynamiquement les contours de l'objet. Si la BBox est trop lâche (elle englobe trop d'arrière-plan), le modèle peut se tromper et inclure des objets adjacents. À l'inverse, si elle est trop serrée et ampute une partie de l'objet, SAM restreint strictement son masque à la zone visible dans la boîte, coupant ainsi logiquement la segmentation.

## 6. Affinage (Points FG/BG)

**Comparaison (Avant / Après) :**
![Image 1 - BBox seule (Echec)](masque_echoue_bbox.png)
![Image 1 - BBox + Points (Réussite)](masque_reussi_points.png)

* **Points utilisés :** `[(250, 190, 1), (280, 210, 0)]` *(1 = Foreground, 0 = Background)*

**Commentaire :**
L'ajout de points de background (BG) devient absolument indispensable lorsque deux objets de même couleur ou de même texture se chevauchent (par exemple, une personne portant un sac à dos de la même couleur que son manteau). La BBox seule échoue souvent à identifier cette subtile frontière. Les clics négatifs (BG) forcent SAM à exclure ces zones spécifiques de son masque. Malgré cela, le modèle échoue encore parfois sur des structures extrêmement fines (câbles, grillages) ou sur des bords floutés par le mouvement, où même une accumulation de points peine à forcer une frontière nette.

## 7. Bilan et réflexion (POC vers produit)

**Analyse des échecs :**
Les limites du modèle se manifestent principalement sous trois facteurs : les occlusions sévères, le manque de contraste (bords fondus avec l'arrière-plan) et la finesse extrême de certains objets. 

**Actions concrètes pour améliorer :**
Pour passer de cette preuve de concept (POC) à un produit robuste, il faudrait d'abord implémenter un prétraitement des images (comme l'égalisation d'histogramme CLAHE) pour aider le modèle sur les zones à faible contraste. Si l'application vise un domaine médical ou industriel très spécifique, un *fine-tuning* de l'encodeur de masque sur un dataset métier serait pertinent. Enfin, côté UI, il faudrait ajouter un outil "pinceau" classique permettant à l'utilisateur de gommer ou peindre manuellement les derniers pourcents d'erreur que SAM n'arrive pas à corriger via des clics.

**Industrialisation (Monitoring & Logs) :**
Si je devais passer cette brique en production, je monitorerais en priorité les métriques suivantes pour garantir la stabilité et la pertinence de l'outil :
* **Temps d'inférence (Latency) :** Pour détecter une éventuelle surcharge du GPU ou un goulot d'étranglement réseau.
* **Score de confiance de SAM :** Une baisse moyenne de l'IoU prédit indique un *drift* sur la nature des images entrantes par rapport à ce que le modèle sait gérer.
* **Nombre de masques modifiés/rejetés par l'utilisateur :** C'est la métrique métier la plus importante, elle indique la pertinence réelle de l'assistance algorithmique.
* **Volume des BBox (ratio bbox/image) :** Pour repérer des anomalies d'utilisation ou des clics aberrants sur l'interface front-end.
* **Erreurs OOM (Out Of Memory) GPU :** Indispensable pour dimensionner correctement l'infrastructure matérielle lors des pics de charge concurrents.