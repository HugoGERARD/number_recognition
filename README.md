# Number Recognition

Projet d'apprentissage ML autour de MNIST : explorer les données, entraîner un CNN PyTorch, mesurer ses performances et prédire un chiffre dessiné à la main.

## Structure

- `src/number_recognition/model.py` : architecture `DigitCNN` et sauvegarde/chargement du checkpoint.
- `src/number_recognition/data.py` : téléchargement MNIST, split train/validation et `DataLoader`.
- `src/number_recognition/training.py` : boucle d'entraînement et métriques train/validation.
- `src/number_recognition/inference.py` : prétraitement d'image et prédiction réutilisables par plusieurs interfaces.
- `src/number_recognition/drawing_app.py` : petite interface Tkinter de dessin, utile comme prototype local.
- `src/number_recognition/api.py` : API HTTP FastAPI avec vérification de santé et prédiction par upload.
- `src/number_recognition/serve.py` : démarrage local de l'API avec le checkpoint entraîné.
- `notebooks/01_train_mnist.ipynb` : exploration, choix des paramètres, entraînement et analyse.
- `notebooks/02_inference_drawing.ipynb` : chargement du modèle et ouverture du prototype de dessin.
- `tests/` : tests unitaires des parties réutilisables.
- `data/` et `models/*.pth` : fichiers locaux ignorés par Git.

## Installation Windows

Depuis PowerShell à la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pour une carte NVIDIA, installe le build CUDA officiel de PyTorch dans ce venv :

```powershell
python -m pip install --no-deps --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Ce wheel est volumineux. Le pilote NVIDIA doit être installé ; le CUDA Toolkit n'a pas besoin d'être installé séparément. Après installation, sélectionne `.venv` comme kernel du notebook et redémarre le kernel.

## Parcours

1. Ouvre `notebooks/01_train_mnist.ipynb` et exécute les cellules dans l'ordre. MNIST est téléchargé sous `data/MNIST/`.
2. Modifie les paramètres d'expérience visibles dans le notebook, puis entraîne le modèle.
3. Exécute la cellule de sauvegarde. Le checkpoint est écrit dans `models/digit_cnn.pth`.
4. Ouvre `notebooks/02_inference_drawing.ipynb` et exécute ses cellules. La fenêtre Tkinter utilise le même prétraitement et le même CNN que le futur backend.

## Découvrir l'API backend

Après avoir entraîné et sauvegardé le modèle, démarre l'API depuis la racine du dépôt :

```powershell
python -m number_recognition.serve
```

L'API écoute sur `http://127.0.0.1:8000`. Ouvre `http://127.0.0.1:8000/docs` pour essayer les routes :

- `GET /health` indique si le backend tourne et quel appareil PyTorch utilise.
- `POST /predict` reçoit une image PNG/JPEG dans le champ `file` et renvoie le chiffre prédit et sa confiance.

La fenêtre Tkinter reste un prototype local. Le futur front navigateur enverra son dessin à cette même logique `predict_image` via l'API. Docker viendra ensuite, quand le contrat entre front et backend sera clair.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

## Git, étape par étape

La refactorisation a été fusionnée dans `main`. L'API est développée sur `feat/inference-api`. Après ses tests et sa pull request, les prochaines branches seront créées depuis le `main` mis à jour :

1. `feat/web-ui` : créer une page de dessin avec prédiction en direct.
2. `chore/docker` : conteneuriser le backend, puis définir comment lancer le front et l'API ensemble.

Pour chaque branche : lancer les tests, examiner le contenu staged, committer, pousser la branche et ouvrir une pull request vers `main` avant de commencer l'étape suivante.

```powershell
git status
git add <fichiers-de-cette-etape>
git diff --cached
git commit -m "type: description courte"
git push -u origin <nom-de-branche>
```

Le remote `origin` pointe vers le dépôt GitHub. Les données de `data/` et les checkpoints `models/*.pth` sont ignorés : ils ne doivent pas être ajoutés aux commits.