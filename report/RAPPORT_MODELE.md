# Rapport technique — Amélioration et évaluation d'un chatbot à LLMs locaux

**Étudiant :** [à renseigner]  
**Enseignante :** Pre. OUHMIDA  
**Module :** Systèmes décisionnels  
**Date de réalisation :** [à renseigner]

## 1. Objectifs

Améliorer l'application initiale Streamlit/Ollama : gestion multi-conversations, comparaison synchronisée des paramètres, performances, surveillance CPU/RAM/GPU, gestion des erreurs et interface.

## 2. Architecture

- **Streamlit (`app.py`)** : interface et `session_state`, sélection des modèles, historiques et export.
- **Ollama (`core.py`)** : appel `Client.chat(..., stream=True)`, paramètres de génération communs, collecte de `eval_count` et `eval_duration`.
- **Surveillance (`monitor.py`)** : échantillonnage `psutil` et `nvidia-smi` en parallèle des appels.
- **Expériences (`questions.json`)** : cinq tâches représentatives envoyées à deux modèles.

## 3. Améliorations fonctionnelles (captures à insérer)

1. [Capture : création/choix/suppression et téléchargement des conversations]
2. [Capture : comparaison côte à côte]
3. [Capture : tableaux et graphiques]
4. [Capture : ressources et matériel détecté]
5. [Capture : message d'erreur Ollama et indicateur de génération]

## 4. Matériel et conditions expérimentales

**CPU :** [nom/modèle/cœurs]  
**RAM :** [Go]  
**GPU / VRAM :** [modèle / Go, ou non disponible]  
**Système et version Python :** [...]  
**Version Ollama :** [...]  
**Modèles (tags exacts) :** [llama3.2:... / qwen2.5:0.5b]  
**Paramètres :** température [...], top-k [...], top-p [...], max tokens [...], seed [...].

**Conditions :** appels séquentiels, applications concurrentes [...], charge préalable [...], nombre de répétitions [...], éventuel échauffement [...].

## 5. Questions

Reprendre ici le contenu de `questions.json` : explication, raisonnement, code Python, résumé, traduction.

## 6. Résultats expérimentaux

Coller le tableau **réel** généré par l'application (CSV `mesures_llm.csv`), puis ajouter les figures exportées/capturées. Prévoir pour chacun des dix essais : temps total, tokens, tokens/s, TTFT, CPU, RAM, GPU et VRAM lorsque disponibles.

| Question | Modèle | Temps (s) | Tokens | Tok/s | TTFT (s) | CPU (%) | RAM max (Go) | GPU (%) | VRAM max (Mo) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | Llama 3.2 | *à mesurer* | | | | | | | |
| Q1 | Qwen 0.5B | *à mesurer* | | | | | | | |
| Q2 | Llama 3.2 | *à mesurer* | | | | | | | |
| Q2 | Qwen 0.5B | *à mesurer* | | | | | | | |
| Q3 | Llama 3.2 | *à mesurer* | | | | | | | |
| Q3 | Qwen 0.5B | *à mesurer* | | | | | | | |
| Q4 | Llama 3.2 | *à mesurer* | | | | | | | |
| Q4 | Qwen 0.5B | *à mesurer* | | | | | | | |
| Q5 | Llama 3.2 | *à mesurer* | | | | | | | |
| Q5 | Qwen 0.5B | *à mesurer* | | | | | | | |

## 7. Analyse critique (à renseigner à partir des résultats)

1. **Temps de réponse le plus faible :** [... données et moyenne ...]
2. **Vitesse de génération la plus élevée :** [... données ...]
3. **Consommation de ressources :** [... CPU/RAM/GPU/VRAM ; tenir compte du caractère global des mesures ...]
4. **Constante selon les tâches ?** [... variations et explications ...]
5. **Recommandation pour machine limitée :** [... compromis mémoire, vitesse, exactitude ...]

**Qualité des réponses :** comparer exactitude du raisonnement, exécution/correction du code, fidélité du résumé, traduction, hallucinations ; éventuellement noter manuellement chaque réponse sur 5. Les performances temporelles seules ne mesurent pas la qualité.

**Limites :** chargement initial, cache, exécutions non répétées, ordre des modèles, ressources mesurées au niveau système, variabilité de la génération, résolution de l'échantillonnage, éventuelle absence de GPU.

## 8. Conclusion

[Conclusion argumentée sur avantages, limites et cas d'usage recommandés. Ne pas conclure sans les mesures réelles.]
