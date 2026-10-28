# Frontend de dessin

Petit frontend statique : le navigateur dessine sur un canvas puis envoie son PNG au backend FastAPI.

## Lancer en développement

1. Dans un terminal, démarre le backend depuis la racine du dépôt :

   ```powershell
   .\.venv\Scripts\python.exe -m number_recognition.serve
   ```

2. Dans un second terminal, démarre ce frontend :

   ```powershell
   .\.venv\Scripts\python.exe -m http.server 5500 --directory frontend
   ```

3. Ouvre `http://127.0.0.1:5500`.

Le navigateur envoie `POST /predict` à `http://127.0.0.1:8000`. Le backend doit donc tourner avant de dessiner.
