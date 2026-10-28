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
- `frontend/` : page web statique avec canvas et prédiction en direct via l'API.
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

## Lancer le backend et le front local

Après avoir entraîné le modèle et exécuté la cellule qui enregistre `models/digit_cnn.pth`, lance l'API dans un terminal :

```powershell
	.\.venv\Scripts\python.exe -m number_recognition.serve
```

Dans un second terminal, démarre le front statique :

```powershell
	.\.venv\Scripts\python.exe -m http.server 5500 --directory frontend
```

Ouvre `http://127.0.0.1:5500` pour dessiner. Le front envoie un PNG à `POST /predict` sur l'API (`http://127.0.0.1:8000`). `http://127.0.0.1:8000/docs` reste disponible pour explorer l'API directement avec Swagger.

L'API autorise actuellement le front local via CORS. Cette configuration de développement devra être ajustée quand le front et l'API seront déployés ou conteneurisés.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

## Git, étape par étape

La refactorisation et l'API sont fusionnées dans `main`. Le front est développé sur `feat/web-ui`. La prochaine étape sera `chore/docker`, après tests et pull request du front :

1. `chore/docker` : conteneuriser le backend, puis définir comment lancer le front et l'API ensemble.

Pour chaque branche : lancer les tests, examiner le contenu staged, committer, pousser la branche et ouvrir une pull request vers `main` avant de commencer l'étape suivante.

```powershell
git status
git add <fichiers-de-cette-etape>
git diff --cached
git commit -m "type: description courte"
git push -u origin <nom-de-branche>
```

Le remote `origin` pointe vers le dépôt GitHub. Les données de `data/` et les checkpoints `models/*.pth` sont ignorés : ils ne doivent pas être ajoutés aux commits.