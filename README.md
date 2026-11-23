# ENSET Chatbot — Amélioration et évaluation de LLMs locaux

TP de **Pre. Asmae OUHMIDA**, cours **Systèmes décisionnels** (ENSET, II BDCC 3 / GLSID 3).

Ce dépôt améliore le chatbot initial Streamlit + Ollama, conservé dans `legacy/ENSET_Chatbot.py`.

## Fonctionnalités

- **Multi-conversations** : créer, sélectionner, supprimer et télécharger l'historique en TXT/JSON, le temps de la session.
- **Comparaison de deux modèles** : même question, mêmes paramètres de génération et réponses côte à côte.
- **Indicateurs mesurés** : temps de réponse, nombre de tokens produits, tokens/s et temps jusqu'au premier token (TTFT).
- **Monitoring matériel** : CPU/RAM avec psutil ; GPU/VRAM avec nvidia-smi si NVIDIA est disponible.
- **Robustesse** : détecter les modèles Ollama installés, signaler les erreurs de connexion/génération et afficher la progression.
- **Benchmark** : cinq questions (explication, raisonnement, code Python, synthèse, traduction) adressées aux deux modèles ; export CSV/JSON et graphiques.

## Installation

### Sur Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull qwen2.5:0.5b
streamlit run app.py
```

### Sur Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull qwen2.5:0.5b
streamlit run app.py
```

Installer et démarrer d'abord [Ollama](https://ollama.com/download). Sur Windows, démarrer l'application Ollama ; si nécessaire, lancer `ollama serve`. Python 3.10+ requis.

**Attention :** le TP demande Llama 3.2 et Qwen 0.5B. Le tag `qwen2.5:0.5b` est proposé comme variante de Qwen 0.5B, et les tags exacts installés doivent être sélectionnés dans l'interface. Llama 3.2 3B demande davantage de mémoire.

## Utilisation

1. **Conversation** : choisir un modèle, démarrer une conversation, utiliser les contrôles d'historique et les boutons d'export dans la barre latérale.
2. **Comparaison des modèles** : sélectionner deux modèles distincts, saisir une question et lancer les générations successives avec les mêmes paramètres.
3. **Expérimentation** : sélectionner Llama 3.2 et Qwen 0.5B, lancer les dix requêtes et exporter les mesures au format CSV/JSON.
4. **Rapport** : compléter `report/RAPPORT_MODELE.md` avec les mesures *réelles*, les captures d'écran, l'analyse des différences et la conclusion.

## Architecture

```text
app.py                        Interface Streamlit et état de session
core.py                       API Ollama, génération en streaming et statistiques
monitor.py                    Monitoring système via psutil et nvidia-smi
questions.json                Cinq questions pour le benchmark
legacy/ENSET_Chatbot.py       Source initiale fournie par l'enseignante
report/RAPPORT_MODELE.md      Rapport technique à compléter
tests/test_core.py            Tests unitaires hors ligne
requirements.txt              Dépendances Python
```

## Mesures et limites

- `time_s` est la durée complète mesurée côté Python, chargement éventuel inclus.
- `generated_tokens` provient de `eval_count` d'Ollama.
- `tokens_s` est calculé avec `eval_count / eval_duration`, une mesure de vitesse de génération côté modèle.
- `ttft_s` mesure l'attente avant le premier fragment de texte non vide.
- CPU, RAM, GPU et VRAM sont des **mesures globales de la machine** ; elles ne représentent pas exclusivement l'utilisation du processus Ollama. L'absence d'un GPU NVIDIA produit « N/D ».
- Les exécutions sont **séquentielles** afin de limiter la concurrence. Le cache, le chargement des modèles, les applications ouvertes et l'ordre des tests peuvent influencer les observations. Pour améliorer la rigueur, répéter et alterner les modèles.
- Les conversations ne sont pas sauvegardées de manière permanente. **Aucun résultat d'expérience n'est simulé ni inventé**.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py core.py monitor.py
```

## Documentation de référence

- [Ollama Python SDK](https://github.com/ollama/ollama-python)
- [Ollama API](https://docs.ollama.com/api)
- [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [psutil](https://psutil.readthedocs.io/)
