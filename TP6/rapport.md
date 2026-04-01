# Rapport TP6 - Explicabilité et MLOps

## 1. Inférence, Grad-CAM et détection de biais

**Images générées :**
![Grad-CAM Normal 1](gradcam_normal_1.png)
![Grad-CAM Normal 2](gradcam_normal_2.png)
![Grad-CAM Pneumo 1](gradcam_pneumo_1.png)

**Performances d'exécution :**
L'inférence pure du modèle ResNet est très rapide (environ 0.13 seconde) et le calcul de la carte d'activation Grad-CAM est quasi instantané (environ 0.12 seconde). Ce surcoût est négligeable pour un déploiement synchrone.

**Analyse des Faux Positifs :**
Lors de l'analyse de l'image `normal_2.jpeg` (une radiographie saine), le modèle s'est trompé en prédisant la classe `PNEUMONIA`. L'explication Grad-CAM montre que le réseau ne regarde pas l'anatomie pulmonaire pour prendre sa décision, mais concentre son attention (zone rouge) sur des artefacts externes à la pathologie, comme les lettrages médicaux sur les bords de l'image. Le modèle a appris un raccourci fallacieux issu des biais du dataset d'entraînement.

**Granularité de l'explication :**
Les zones mises en évidence ressemblent à de gros blocs très flous. Cette perte de résolution spatiale est due à l'architecture même des réseaux convolutifs (ResNet). Grad-CAM s'applique sur la toute dernière couche (feature map), là où les opérations successives de *pooling* ont réduit l'image à une très petite grille (ex: 7x7). Le redimensionnement (upsampling) de cette grille à la taille de l'image d'origine crée cet effet pixelisé et imprécis.

---

## 2. Integrated Gradients (IG) et SmoothGrad

**Image générée :**
![Integrated Gradients et SmoothGrad](ig_smooth_normal_1.png)

**Temps d'exécution et architecture MLOps temps réel :**
Nos logs montrent une explosion du temps de calcul pour l'explicabilité fine :
* **Temps IG pur :** 10.10 secondes
* **Temps SmoothGrad (IG x 100) :** 779.95 secondes (soit près de 13 minutes).

Il est technologiquement impossible de générer l'explication SmoothGrad de manière synchrone (en temps réel) lorsqu'un médecin clique pour analyser une radio. L'architecture logicielle adéquate impose un découplage : l'API doit renvoyer instantanément la prédiction clinique, puis déléguer le calcul lourd de l'explicabilité à une file d'attente asynchrone (message broker type RabbitMQ/Kafka avec des *workers* Celery). Le résultat sera poussé sur l'interface (via WebSockets) une fois terminé.

**Avantage mathématique du négatif :**
Grad-CAM utilise un filtre ReLU qui masque les valeurs négatives. Integrated Gradients, en revanche, conserve ces valeurs (visibles en bleu sur la carte). L'avantage est d'obtenir une transparence totale : on identifie les pixels qui soutiennent la prédiction (positifs), mais aussi ceux qui s'y opposent mathématiquement (négatifs), ce qui permet de comprendre ce qui a fait "hésiter" le modèle.

---

## 3. Modélisation Intrinsèquement Interprétable (Glass-box)

**Image générée :**
![Coefficients Régression Logistique](glassbox_coefficients.png)

**Performance :**
* Accuracy obtenue sur les données de test : **97.37%**

**Impact des caractéristiques pour la classe "Maligne" :**
Sur le graphique de la Régression Logistique, les coefficients négatifs (barres rouges) tirent la prédiction vers la classe 0 (Maligne). La caractéristique ayant le plus d'impact est celle possédant la barre rouge la plus longue (valeur absolue la plus élevée), ce qui correspond aux dimensions extrêmes des cellules mesurées (comme `worst radius`, `worst perimeter` ou `worst area`).

**Avantage d'un modèle interprétable :**
L'avantage majeur d'un modèle "Glass-box" est que sa logique est directement et exactement lisible via ses propres paramètres d'apprentissage (les poids β). Contrairement aux méthodes post-hoc pour les boîtes noires, il n'y a pas besoin d'un algorithme d'approximation tiers qui pourrait introduire des biais d'interprétation.

---

## 4. Explicabilité Post-Hoc avec SHAP sur un Modèle Complexe

**Images générées :**
![SHAP Summary Plot](shap_summary.png)
![SHAP Waterfall Plot](shap_waterfall.png)

**Performance de la boîte noire :**
* Accuracy du Random Forest : **96.49%**

**Explicabilité globale :**
En comparant le *Summary Plot* de SHAP avec le graphique de la Régression Logistique, on observe que le Top 3 des variables les plus importantes est identique entre les deux modèles. On en déduit que ces caractéristiques cliniques sont des biomarqueurs extrêmement robustes, puisqu'ils sont considérés comme primordiaux par deux algorithmes aux mathématiques radicalement différentes (linéaire vs ensembliste non-linéaire).

**Explicabilité locale :**
Sur l'explication spécifique au patient 0, la caractéristique ayant le plus contribué à tirer la prédiction vers sa valeur finale est représentée par la barre la plus large. La valeur numérique exacte mesurée chez ce patient est directement lisible sur l'axe de gauche du graphique.