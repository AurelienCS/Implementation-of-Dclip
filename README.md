# Efficiently Disentangling CLIP For MultiObject Perception (Implementation)

**Auteurs :** Gautier Cai, Aurélien Chambolle-Solaz, Guillaume Volland

## Introduction

Cette implémentation PyTorch se base sur le papier *"Efficiently Disentangling CLIP for Multi-Object Perception"*. Le projet vise à améliorer les performances du modèle CLIP dans des tâches de reconnaissance multi-objets, où le modèle standard échoue souvent en raison de l'enchevêtrement des caractéristiques visuelles et textuelles.

L'approche D-CLIP introduit deux mécanismes de régularisation appliqués à un backbone CLIP figé :
1.  **Mutual Fisher Information (MFI) Loss :** Une contrainte qui régule la matrice de similarité pour minimiser les corrélations inter-classes et favoriser le désenchevêtrement sémantique.
2.  **Asymmetric Loss (ASL) :** Une fonction de perte optimisée pour l'apprentissage multi-label, alignant les caractéristiques visuelles avec les projecteurs textuels.

## Structure du Dépôt

Le code est organisé de la manière suivante :

* `data/` : Scripts de gestion des jeux de données (VOC2007, COCO).
* `models/` : Définition de l'architecture D-CLIP et des projecteurs.
* `weights/` : Dossier de stockage pour les poids des modèles entraînés.
* `results/` : Logs d'entraînement et visualisations (heatmaps).
* `notebooks/` : Notebooks Jupyter pour l'exploration et la démo.
* `dclip.py` : Module principal contenant la classe `DClip`.
* `train.py` : Script d'entraînement et boucle d'optimisation.
* `eval.py` : Script d'évaluation pour le calcul du mAP.

## Installation et Dépendances

L'environnement nécessite Python 3.8+ et les bibliothèques suivantes :

```bash
pip install torch torchvision clip tqdm pandas scikit-learn
