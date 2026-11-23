"""ENSET Chatbot — TP Améliorations LLMs (Pr. OUHMIDA).

Run: streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from core import GenerationOptions, build_client, generate, installed_models
from monitor import hardware_info


st.set_page_config(page_title="ENSET Chatbot — LLMs locaux", page_icon="🧠", layout="wide")
st.title("🧠 ENSET Chatbot")
st.caption("TP : amélioration et évaluation de modèles locaux — Pr. OUHMIDA")

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def new_conversation() -> str:
    ident = uuid.uuid4().hex[:12]
    st.session_state.conversation_counter += 1
    st.session_state.conversations[ident] = {
        "title": f"Conversation {st.session_state.conversation_counter}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    st.session_state.active_id = ident
    return ident


def init_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "conversation_counter" not in st.session_state:
        st.session_state.conversation_counter = 0
    if "active_id" not in st.session_state or st.session_state.active_id not in st.session_state.conversations:
        new_conversation()
    if "comparison" not in st.session_state:
        st.session_state.comparison = []
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = []
    if "benchmark_metadata" not in st.session_state:
        st.session_state.benchmark_metadata = {}


def history_as_txt(conv: dict) -> str:
    text = [conv["title"], f"Création : {conv['created_at']}", ""]
    for message in conv["messages"]:
        role = "Utilisateur" if message["role"] == "user" else "Assistant"
        text.append(f"{role} :\n{message['content']}\n")
    return "\n".join(text)


def format_metric(x, decimals: int = 2, unit: str = "") -> str:
    if x is None:
        return "N/D"
    return f"{x:.{decimals}f}{unit}" if isinstance(x, (int, float)) else str(x)


def show_results(results: list[dict]) -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    columns = [
        "question_type", "question_id", "model", "time_s", "generated_tokens",
        "tokens_s", "ttft_s", "cpu_avg_pct", "ram_peak_gb", "gpu_avg_pct",
        "vram_peak_mb",
    ]
    shown = df[[c for c in columns if c in df.columns]].copy()
    st.dataframe(shown, width="stretch", hide_index=True)
    for key, label in (("time_s", "Temps total (s)"),
                       ("tokens_s", "Tokens/seconde"),
                       ("cpu_avg_pct", "CPU moyen (%)"),
                       ("ram_peak_gb", "RAM max utilisée (Go)"),
                       ("gpu_avg_pct", "GPU moyen (%)"),
                       ("vram_peak_mb", "VRAM max utilisée (Mo)")):
        if key in df.columns and df[key].notna().any():
            with st.expander(f"Graphique : {label}", expanded=key in ("time_s", "tokens_s")):
                chart = df.assign(test=df.get("question_id", pd.Series(["Question"] * len(df))).astype(str))
                st.bar_chart(chart, x="test", y=key, color="model", x_label="Question", y_label=label)


def comparison_pair(models: list[str], key_prefix: str) -> tuple[str | None, str | None]:
    a, b = st.columns(2)
    with a:
        first = st.selectbox("Modèle A", options=models, key=f"{key_prefix}_first")
    with b:
        second = st.selectbox("Modèle B", options=models,
                              index=1 if len(models) > 1 else 0, key=f"{key_prefix}_second")
    if first == second:
        st.info("Choisissez deux modèles différents pour comparer les résultats.")
    return first, second


init_state()

with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Mode", ["Conversation", "Comparaison des modèles", "Expérimentation"],
                    label_visibility="collapsed")
    host = st.text_input("Adresse Ollama", value="http://localhost:11434",
                         help="Service Ollama local (ou une autre URL Ollama de confiance).")
    temperature = st.slider("Température", 0.0, 2.0, 0.7, 0.1)
    top_k = st.slider("Top-k", 1, 100, 40)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Nombre maximal de tokens", 1, 2048, 512)
    seed = st.number_input("Seed (reproductibilité)", 0, 2_147_483_647, 42)

opts = GenerationOptions(temperature=temperature, top_k=top_k, top_p=top_p,
                         num_predict=max_tokens, seed=int(seed))
try:
    client = build_client(host)
    models = installed_models(client)
    connection_error = None
except Exception as exc:
    models = []
    connection_error = str(exc)

if connection_error:
    st.error("Le service Ollama est inaccessible. Vérifiez qu'Ollama est installé et lancé. "
             f"Adresse : `{host}`. Détail : {connection_error}")
    st.code("ollama serve", language="bash")
    st.stop()
if not models:
    st.warning("Ollama est accessible, mais aucun modèle local n'est installé. "
               "Téléchargez d'abord les modèles avec :")
    st.code("ollama pull llama3.2\nollama pull qwen2.5:0.5b", language="bash")
    st.stop()

with st.sidebar:
    st.success(f"Ollama connecté · {len(models)} modèle(s)")
    with st.expander("Modèles disponibles"):
        for name in models:
            st.caption(name)

if mode == "Conversation":
    with st.sidebar:
        st.divider()
        st.subheader("Conversations")
        if st.button("➕ Nouvelle conversation", width="stretch"):
            new_conversation()
            st.rerun()
        ids = list(st.session_state.conversations)
        chosen = st.selectbox("Historique", ids, index=ids.index(st.session_state.active_id),
                              format_func=lambda i: st.session_state.conversations[i]["title"])
        if chosen != st.session_state.active_id:
            st.session_state.active_id = chosen
            st.rerun()
        if st.button("🗑️ Supprimer cette conversation", width="stretch"):
            del st.session_state.conversations[st.session_state.active_id]
            if st.session_state.conversations:
                st.session_state.active_id = next(iter(st.session_state.conversations))
            else:
                new_conversation()
            st.rerun()

        conv = st.session_state.conversations[st.session_state.active_id]
        st.download_button("⬇️ Exporter TXT", history_as_txt(conv),
                           file_name=f"{st.session_state.active_id}.txt", mime="text/plain",
                           width="stretch")
        st.download_button("⬇️ Exporter JSON", json.dumps(conv, ensure_ascii=False, indent=2),
                           file_name=f"{st.session_state.active_id}.json",
                           mime="application/json", width="stretch")
        selected_model = st.selectbox("Modèle de conversation", models)

    st.subheader(conv["title"])
    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("metrics"):
                with st.expander("Performances"):
                    st.json(msg["metrics"])

    prompt = st.chat_input("Posez votre question au modèle local...")
    if prompt:
        if not conv["messages"]:
            conv["title"] = prompt.strip().replace("\n", " ")[:40] or conv["title"]
        conv["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Génération de la réponse..."):
                try:
                    result = generate(client, selected_model, conv["messages"], opts,
                                      on_update=lambda text: placeholder.markdown(text + " ▌"))
                except Exception as exc:
                    st.error(f"Erreur de génération : {exc}")
                else:
                    placeholder.markdown(result.response or "*(Réponse vide)*")
                    conv["messages"].append({"role": "assistant", "content": result.response,
                                              "model": selected_model, "metrics": result.as_dict()})
                    st.caption(f"{result.time_s:.2f} s · "
                               f"{result.generated_tokens if result.generated_tokens is not None else 'N/D'} tokens · "
                               f"{format_metric(result.tokens_s)} tok/s · "
                               f"TTFT {format_metric(result.ttft_s)} s")

elif mode == "Comparaison des modèles":
    st.subheader("⚖️ Comparaison de deux modèles")
    if len(models) < 2:
        st.warning("Au moins deux modèles installés sont nécessaires pour ce mode.")
        st.stop()
    first, second = comparison_pair(models, "compare")
    prompt = st.text_area("Question identique envoyée aux deux modèles", height=110,
                          placeholder="Expliquez la différence entre ETL et ELT...")
    if st.button("Comparer les réponses", type="primary", disabled=not prompt.strip() or first == second):
        st.session_state.comparison = []
        for model in (first, second):
            with st.spinner(f"Génération en cours : {model}"):
                try:
                    result = generate(client, model, [{"role": "user", "content": prompt}], opts)
                except Exception as exc:
                    st.error(f"Erreur sur {model} : {exc}")
                else:
                    st.session_state.comparison.append(result.as_dict())
    if st.session_state.comparison:
        columns = st.columns(len(st.session_state.comparison))
        for col, result in zip(columns, st.session_state.comparison):
            with col:
                st.markdown(f"### {result['model']}")
                st.markdown(result["response"] or "*(Réponse vide)*")
                st.caption(f"{format_metric(result['time_s'])} s | "
                           f"{format_metric(result['tokens_s'])} tok/s | "
                           f"TTFT : {format_metric(result['ttft_s'])} s")
        st.subheader("Tableau des performances")
        show_results(st.session_state.comparison)
        st.download_button("Exporter la comparaison JSON",
                           json.dumps({"question": prompt, "options": opts.as_dict(),
                                       "results": st.session_state.comparison},
                                      ensure_ascii=False, indent=2),
                           file_name="comparaison.json", mime="application/json")

else:
    st.subheader("🔬 Expérimentation — cinq tâches variées")
    st.caption("Chaque question est envoyée séparément et séquentiellement aux deux modèles "
               "avec les mêmes paramètres. Les mesures sont collectées sur votre machine réelle.")
    with st.expander("Questions de l'expérience", expanded=True):
        for q in QUESTIONS:
            st.markdown(f"**{q['id']} — {q['type']}** : {q['prompt']}")
    if len(models) < 2:
        st.warning("Installez deux modèles pour lancer l'expérience.")
        st.stop()
    first, second = comparison_pair(models, "bench")
    if not any("llama3.2" in m.lower() for m in (first, second)) or not any(
        "qwen" in m.lower() and "0.5b" in m.lower() for m in (first, second)
    ):
        st.warning("Le sujet demande spécifiquement Llama 3.2 et Qwen 0.5B. "
                   "Vérifiez les modèles choisis avant la collecte finale.")
    if st.button("Lancer les 10 générations", type="primary", disabled=first == second):
        rows = []
        progress = st.progress(0, text="Initialisation...")
        failures = []
        for i, q in enumerate(QUESTIONS):
            for model in (first, second):
                progress.progress((len(rows) + len(failures)) / (len(QUESTIONS) * 2),
                                  text=f"{q['id']} · {model}")
                try:
                    result = generate(client, model, [{"role": "user", "content": q["prompt"]}], opts)
                except Exception as exc:
                    failures.append(f"{q['id']} / {model} : {exc}")
                    continue
                rows.append({"question_id": q["id"], "question_type": q["type"],
                             "question": q["prompt"], **result.as_dict()})
        progress.progress(1.0, text="Expérimentation terminée")
        st.session_state.benchmark_results = rows
        st.session_state.benchmark_metadata = {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "host": host, "models": [first, second], "options": opts.as_dict(),
            "hardware": hardware_info(), "conditions": "Exécutions séquentielles; mesures système globales",
            "errors": failures,
        }
        if failures:
            st.warning("Certaines générations ont échoué :\n" + "\n".join(failures))
    if st.session_state.benchmark_results:
        rows = st.session_state.benchmark_results
        st.subheader("Résultats observés")
        show_results(rows)
        summary = pd.DataFrame(rows).groupby("model")[["time_s", "tokens_s"]].mean(numeric_only=True)
        st.markdown("**Moyennes descriptives (sur les exécutions disponibles)**")
        st.dataframe(summary, width="stretch")
        payload = {"metadata": st.session_state.benchmark_metadata, "results": rows}
        st.download_button("⬇️ Rapport de mesures JSON", json.dumps(payload, ensure_ascii=False, indent=2),
                           file_name="mesures_llm.json", mime="application/json")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("⬇️ Tableau CSV", buffer.getvalue().encode("utf-8-sig"),
                           file_name="mesures_llm.csv", mime="text/csv")
        st.info("Pour le rapport, évaluez aussi manuellement la qualité des réponses; les "
                "indicateurs de vitesse ne mesurent pas l'exactitude.")

with st.sidebar:
    st.divider()
    with st.expander("Matériel détecté"):
        if st.button("Actualiser les informations matérielles"):
            st.json(hardware_info())
    st.caption("Conversations conservées uniquement durant cette session Streamlit.")
