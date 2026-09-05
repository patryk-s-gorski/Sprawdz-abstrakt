import json
import os
from datetime import datetime

import streamlit as st
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "criteria_config.json")

GOTOWOSC_STYLE = {
    "gotowy": ("success", "Gotowy do wysłania"),
    "drobne_poprawki": ("warning", "Wymaga drobnych poprawek"),
    "wymaga_pracy": ("error", "Wymaga jeszcze pracy"),
}


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (FileNotFoundError, KeyError):
            api_key = None
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def build_system_prompt(konferencja, kategoria, limit_znakow, jezyk, kryteria):
    kryteria_txt = "\n".join(f"- {k}" for k in kryteria)
    return f"""Jesteś asystentem pomagającym studentom medycyny przygotować abstrakt \
przed wysłaniem go do komitetu recenzenckiego konferencji {konferencja}. \
NIE wystawiasz oceny liczbowej ani werdyktu accept/reject — to wyłącznie rola komitetu \
recenzenckiego. Twoim celem jest pomóc studentowi poprawić tekst przed formalną recenzją.

Abstrakt jest napisany w języku: {jezyk}.
Kategoria zgłoszenia: {kategoria}.

Sprawdź abstrakt pod kątem kryteriów specyficznych dla tej kategorii:
{kryteria_txt}

Sprawdź też ogólnie:
- Jasność języka dla odbiorcy interdyscyplinarnego (medycyna + inżynieria)
- Brak nieuzasadnionego żargonu bez wyjaśnienia
- Spójność logiczna między poszczególnymi częściami
- Jeśli abstrakt jest po angielsku: poprawność gramatyczną i naturalność sformułowań

Limit długości: {limit_znakow} znaków.

Nie licz i nie podawaj długości tekstu w znakach — ta wartość jest liczona precyzyjnie \
po stronie aplikacji, nie przez Ciebie.

Odpowiedz WYŁĄCZNIE poprawnym obiektem JSON, bez żadnego tekstu przed ani po, bez \
znaczników markdown, w dokładnie takiej strukturze (wszystkie teksty po polsku, \
niezależnie od języka abstraktu):

{{
  "gotowosc": "gotowy" | "drobne_poprawki" | "wymaga_pracy",
  "mocne_strony": ["punkt 1", "punkt 2"],
  "do_poprawy": [
    {{"fragment": "krótki cytat z oryginału", "sugestia": "konkretna propozycja zmiany"}}
  ]
}}
"""


def call_claude(client, tekst, konferencja, kategoria, limit_znakow, jezyk, kryteria):
    system_prompt = build_system_prompt(konferencja, kategoria, limit_znakow, jezyk, kryteria)
    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        # Sonnet 5 ma domyślnie włączone adaptacyjne myślenie (bloki "thinking" przed
        # tekstem) i odrzuca niedomyślne parametry samplingu (temperature/top_p/top_k).
        # Ten task to prosta, ustrukturyzowana ocena — wyłączamy myślenie: taniej,
        # szybciej, a odpowiedź zawiera tylko blok tekstowy.
        thinking={"type": "disabled"},
        system=system_prompt,
        messages=[{"role": "user", "content": tekst}],
    )
    # Nie zakładamy pozycji bloku — nawet z wyłączonym thinking to najbardziej odporny wariant.
    for block in message.content:
        if block.type == "text":
            return block.text
    raise ValueError("Odpowiedź modelu nie zawiera bloku tekstowego")


def parse_response(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Brak obiektu JSON w odpowiedzi modelu")
    return json.loads(text[start:end + 1])


def render_wynik(wynik, dlugosc_aktualna, limit_znakow):
    status, etykieta = GOTOWOSC_STYLE.get(
        wynik.get("gotowosc"), ("info", wynik.get("gotowosc", "—"))
    )
    getattr(st, status)(f"**{etykieta}**")
    if dlugosc_aktualna > limit_znakow:
        st.caption(f"⚠️ Długość: {dlugosc_aktualna}/{limit_znakow} znaków — przekracza limit")
    else:
        st.caption(f"Długość: {dlugosc_aktualna}/{limit_znakow} znaków — mieści się")

    if wynik.get("mocne_strony"):
        st.markdown("**Co działa dobrze**")
        for punkt in wynik["mocne_strony"]:
            st.markdown(f"- {punkt}")

    if wynik.get("do_poprawy"):
        st.markdown("**Do poprawy**")
        for i, item in enumerate(wynik["do_poprawy"], 1):
            skrot = item.get("fragment", "")[:60]
            with st.expander(f"{i}. {skrot}..."):
                st.markdown(f"> {item.get('fragment', '')}")
                st.markdown(f"**Sugestia:** {item.get('sugestia', '')}")


def wynik_do_markdown(wynik, konferencja, kategoria, dlugosc_aktualna, limit_znakow):
    etykieta = GOTOWOSC_STYLE.get(
        wynik.get("gotowosc"), (None, wynik.get("gotowosc", "—"))
    )[1]
    lines = [
        f"# Ocena abstraktu — {konferencja} ({kategoria})",
        f"**Status:** {etykieta}",
        "",
        "## Co działa dobrze",
    ]
    lines += [f"- {p}" for p in wynik.get("mocne_strony", [])]
    lines += ["", "## Do poprawy"]
    for item in wynik.get("do_poprawy", []):
        lines.append(f"- **Fragment:** {item.get('fragment', '')}")
        lines.append(f"  **Sugestia:** {item.get('sugestia', '')}")
    status_dlugosci = "mieści się" if dlugosc_aktualna <= limit_znakow else (
        f"przekracza limit o {dlugosc_aktualna - limit_znakow} znaków"
    )
    lines += ["", f"## Długość: {dlugosc_aktualna} / {limit_znakow} znaków ({status_dlugosci})"]
    return "\n".join(lines)


def clear_abstrakt():
    st.session_state.abstrakt_tekst = ""


# ---------------- UI ----------------

st.set_page_config(page_title="Sprawdź swój abstrakt", page_icon="📝", layout="centered")

config = load_config()
client = get_client()

st.title("📝 Sprawdź swój abstrakt")
st.caption("Wstępna, nieformalna ocena przed wysłaniem do komitetu recenzenckiego")

if client is None:
    st.error(
        "Brak klucza API. Ustaw zmienną środowiskową ANTHROPIC_API_KEY "
        "(plik .env lokalnie) lub dodaj ją w Streamlit Secrets przy hostingu."
    )
    st.stop()

if "historia" not in st.session_state:
    st.session_state.historia = []
if "abstrakt_tekst" not in st.session_state:
    st.session_state.abstrakt_tekst = ""

with st.sidebar:
    st.header("Ustawienia")
    nazwy_konferencji = list(config["konferencje"].keys())
    konferencja = st.selectbox("Konferencja", nazwy_konferencji)
    ustawienia_konf = config["konferencje"][konferencja]

    kategoria = st.selectbox("Kategoria zgłoszenia", ustawienia_konf["kategorie"])
    limit_znakow = st.number_input(
        "Limit znaków", min_value=200, max_value=10000,
        value=ustawienia_konf["limit_znakow"], step=100,
    )
    jezyk = st.radio("Język abstraktu", ["Polski", "English"], horizontal=True)

    kryteria = config["kryteria_kategorii"][kategoria]
    with st.expander("Kryteria oceny dla tej kategorii"):
        for k in kryteria:
            st.markdown(f"- {k}")

col_a, col_b = st.columns([4, 1])
with col_a:
    tekst = st.text_area(
        "Wklej treść abstraktu", height=280,
        key="abstrakt_tekst", placeholder="Wklej tutaj swój abstrakt...",
    )
with col_b:
    st.write("")
    st.write("")
    st.button("Wyczyść", use_container_width=True, on_click=clear_abstrakt)

aktualna_dlugosc = len(tekst)
procent = min(aktualna_dlugosc / limit_znakow, 1.0) if limit_znakow else 0.0
st.progress(procent)
kolor = "red" if aktualna_dlugosc > limit_znakow else "gray"
st.caption(f":{kolor}[{aktualna_dlugosc} / {limit_znakow} znaków]")

if st.button("Sprawdź abstrakt", type="primary", disabled=not tekst.strip()):
    raw = None
    with st.spinner(f"Analizuję abstrakt pod kątem kryteriów: {kategoria}..."):
        try:
            raw = call_claude(client, tekst, konferencja, kategoria, limit_znakow, jezyk, kryteria)
        except APIError as e:
            st.error(f"Błąd API Anthropic: {e}")
        except Exception as e:
            st.error(f"Nieoczekiwany błąd połączenia: {e}")

    if raw:
        st.divider()
        wynik = None
        try:
            wynik = parse_response(raw)
        except (ValueError, json.JSONDecodeError):
            st.warning("Nie udało się przetworzyć ustrukturyzowanej odpowiedzi — surowy wynik:")
            st.markdown(raw)

        if wynik:
            render_wynik(wynik, aktualna_dlugosc, limit_znakow)
            raport = wynik_do_markdown(
                wynik, konferencja, kategoria, aktualna_dlugosc, limit_znakow
            )
            st.download_button(
                "Pobierz ocenę (.md)", raport,
                file_name=f"ocena_abstraktu_{datetime.now():%Y%m%d_%H%M}.md",
            )
            st.session_state.historia.append({
                "Godzina": datetime.now().strftime("%H:%M:%S"),
                "Kategoria": kategoria,
                "Status": GOTOWOSC_STYLE.get(
                    wynik.get("gotowosc"), (None, wynik.get("gotowosc"))
                )[1],
                "Długość": f"{aktualna_dlugosc}/{limit_znakow}",
            })

if st.session_state.historia:
    st.divider()
    with st.expander(f"Historia tej sesji ({len(st.session_state.historia)})"):
        st.dataframe(st.session_state.historia, use_container_width=True, hide_index=True)
