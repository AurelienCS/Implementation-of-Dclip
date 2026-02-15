# Efficiently Disentangling CLIP For MultiObject Perception (Implementation)

**Auteurs :** Gautier Cai, Aurélien Chambolle-Solaz, Guillaume Volland  

## Introduction

Cette implémentation PyTorch se base sur le papier *"Efficiently Disentangling CLIP for Multi-Object Perception"*. Le projet vise à améliorer les performances du modèle CLIP dans des tâches de reconnaissance multi-objets, où le modèle standard échoue souvent en raison de l'enchevêtrement des caractéristiques visuelles et textuelles dans l'espace latent.

L'approche D-CLIP introduit deux mécanismes de régularisation appliqués à un backbone CLIP figé :
1. **Mutual Feature Information (MFI) Loss :** Régule la matrice de similarité pour minimiser les corrélations inter-classes et favoriser le désenchevêtrement sémantique.
2. **Asymmetric Loss (ASL) :** Aligne les caractéristiques visuelles avec les représentations textuelles démêlées.

## Structure du Dépôt

Le dépôt est constitué des fichiers suivants :

* `dclip.py` : Architecture du modèle `DClip`, incluant les projecteurs et le mécanisme d'agrégation des scores par attention.
* `train.py` : Script principal gérant la boucle d'entraînement et l'optimisation jointe (ASL + MFI).
* `eval.py` : Fonctions d'évaluation (mAP) et implémentation des fonctions de perte (AsymmetricLossOptimized, MFILossOpti).
* `utils.py` : Utilitaires pour le chargement de CLIP et le patching de la résolution des ViT pour supporter une entrée en 448px.
* `Exemple_Exp.ipynb` : Notebook documentant les étapes d'expérimentation et les sorties d'entraînement.

## Installation

```bash
pip install torch torchvision clip tqdm pandas scikit-learn

```

## Utilisation

L'entraînement s'effectue via la fonction `train_dclip` en spécifiant le backbone souhaité :

```python
from dclip import dclip
from train import train_dclip

# Initialisation du modèle avec les classes VOC2007
model = dclip(classnames=VOC2007_CLASSES, clip_type="ViT-B/32")

# Lancement de l'entraînement
train_dclip(
    model, 
    train_loader, 
    test_loader, 
    class_names=VOC2007_CLASSES, 
    num_epochs=30, 
    lr=0.002, 
    alpha=7e-5
)

```

## Résultats et Analyse

Les tests menés sur PASCAL VOC 2007 présentent les performances suivantes (résultats moyennés sur 5 runs) :

| Modèle | Backbone | MFI | Temps d'entraînement | mAP |
| --- | --- | --- | --- | --- |
| CLIP Standard | ViT-B/32 | Non | 11 min | 0.9093 |
| **D-CLIP** | **ViT-B/32** | Oui | **11 min** | **0.9172** |
| **D-CLIP** | **ViT-B/16** | Oui | **49 min** | **0.9322** |

L'ajout de la régularisation MFI sur ViT-B/32 permet un gain de +0,01 mAP. Le passage au ViT-B/16 améliore significativement la précision grâce à une résolution de patchs plus fine (plus de tokens), bien que cela multiplie le coût computationnel par quatre.

## Conclusion et Limites du Papier Original

L'implémentation confirme l'efficacité de D-CLIP (93,2% mAP sur ViT-B/16). L'utilisation d'un ResNet pourrait potentiellement surpasser ces scores : sa nature convolutive capture mieux les motifs locaux que les patchs du ViT, affinant ainsi l'attention.

Plusieurs incohérences ont été relevées dans le papier original (ArXiv, 2025) :

* **Vitesse :** Les auteurs rapportent un temps d'entraînement identique (~3h) pour ViT-B/16 et ViT-B/32, ce qui contredit nos observations sur l'écart de complexité.
* **Stagnation :** Le mAP reste constant (94,6%) sur ViT-B/16, 32 et RN50 dans le papier, alors que la résolution supérieure du ViT-B/16 devrait normalement accroître la performance.

## Références

1. *Efficiently Disentangling CLIP for Multi-Object Perception* (2025).
2. *PositiveCoOp: Bridging Positive-Negative Gaps in Contrastive Language-Image Pre-training*.
