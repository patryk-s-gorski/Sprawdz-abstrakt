import json
import math
import os
from datetime import datetime

import streamlit as st
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "criteria_config.json")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.jpg")

# Jednostki liczenia limitu. Klucze są stałe (używane w kodzie), etykiety tłumaczone przez t().
JEDNOSTKI = {
    "znaki": {"domyslny_limit": 2000, "min": 200, "max": 10000, "krok": 100},
    "slowa": {"domyslny_limit": 300, "min": 30, "max": 2000, "krok": 10},
}

WYMIARY = ["punktacja_formalna", "punktacja_merytoryczna", "punktacja_jezykowa"]

# ---------------- Tłumaczenia interfejsu ----------------
# Uwaga: kategorie zgłoszeń i kryteria oceny pochodzą z criteria_config.json
# i NIE są tu tłumaczone — to treść konfigurowana przez organizatorów, nie tekst UI.

TRANSLATIONS = {
    "Polski": {
        "app_title": "Sprawdź swój abstrakt",
        "app_caption": "Wstępna, nieformalna ocena przed wysłaniem do komitetu recenzenckiego",
        "access_code_label": "Kod dostępu",
        "enter_button": "Wejdź",
        "wrong_code": "Nieprawidłowy kod dostępu.",
        "no_api_key": "Brak klucza API. Ustaw zmienną środowiskową ANTHROPIC_API_KEY "
                       "(plik .env lokalnie) lub dodaj ją w Streamlit Secrets przy hostingu.",
        "logout": "Wyloguj",
        "settings_header": "Ustawienia",
        "category_label": "Kategoria zgłoszenia",
        "unit_label": "Licz limit w:",
        "unit_znaki": "Znaki",
        "unit_slowa": "Słowa",
        "unit_znaki_sufiks": "znaków",
        "unit_slowa_sufiks": "słów",
        "limit_label": "Limit treści ({jednostka})",
        "abstract_lang_label": "Język abstraktu",
        "lang_polish": "Polski",
        "lang_english": "English",
        "criteria_expander": "Kryteria oceny dla tej kategorii",
        "structure_prefix": "Struktura",
        "structure_sections": "wymagane wyodrębnione sekcje: {sekcje}",
        "structure_continuous": "tekst ciągły, bez wymaganych podnagłówków sekcji",
        "merit_criteria_header": "Kryteria merytoryczne",
        "formal_criteria_header": "Kryteria formalne",
        "title_input_label": "Tytuł abstraktu",
        "title_placeholder": "Wpisz tytuł pracy...",
        "text_area_label": "Wklej treść abstraktu",
        "text_area_placeholder": "Wklej tutaj treść swojego abstraktu (bez tytułu)...",
        "clear_button": "Wyczyść",
        "length_caption": "Treść: {aktualna} / {limit} {jednostka}",
        "check_button": "Sprawdź abstrakt",
        "fill_both_caption": "Wpisz tytuł i treść abstraktu, żeby odblokować sprawdzenie.",
        "analyzing": "Analizuję abstrakt pod kątem kryteriów: {kategoria}...",
        "api_error": "Błąd API Anthropic: {e}",
        "unexpected_error": "Nieoczekiwany błąd połączenia: {e}",
        "parse_warning": "Nie udało się przetworzyć ustrukturyzowanej odpowiedzi — surowy wynik:",
        "download_button": "Pobierz ocenę (.md)",
        "history_expander": "Historia tej sesji ({n})",
        "dim_punktacja_formalna": "Formalna",
        "dim_punktacja_merytoryczna": "Merytoryczna",
        "dim_punktacja_jezykowa": "Językowa",
        "no_score": "Brak wyniku",
        "diagnostic_caption": "Wyniki pomocnicze, diagnostyczne — nie są oceną komitetu recenzenckiego.",
        "length_ok": "Długość treści: {aktualna}/{limit} {jednostka} — mieści się",
        "length_over": "⚠️ Długość treści: {aktualna}/{limit} {jednostka} — przekracza limit",
        "strengths_header": "Co działa dobrze",
        "improve_header": "Do poprawy",
        "suggestion_label": "Sugestia:",
        "status_ready": "Gotowy do wysłania",
        "status_minor": "Wymaga drobnych poprawek",
        "status_needs_work": "Wymaga jeszcze pracy",
        "average_caption": "**{etykieta}** (średnia z 3 wymiarów: {srednia}/100)",
    },
    "English": {
        "app_title": "Check your abstract",
        "app_caption": "A preliminary, informal review before submission to the review committee",
        "access_code_label": "Access code",
        "enter_button": "Enter",
        "wrong_code": "Incorrect access code.",
        "no_api_key": "Missing API key. Set the ANTHROPIC_API_KEY environment variable "
                       "(.env file locally) or add it to Streamlit Secrets when hosting.",
        "logout": "Log out",
        "settings_header": "Settings",
        "category_label": "Submission category",
        "unit_label": "Count limit in:",
        "unit_znaki": "Characters",
        "unit_slowa": "Words",
        "unit_znaki_sufiks": "characters",
        "unit_slowa_sufiks": "words",
        "limit_label": "Body limit ({jednostka})",
        "abstract_lang_label": "Abstract language",
        "lang_polish": "Polish",
        "lang_english": "English",
        "criteria_expander": "Evaluation criteria for this category",
        "structure_prefix": "Structure",
        "structure_sections": "required distinct sections: {sekcje}",
        "structure_continuous": "continuous text, no required subheadings",
        "merit_criteria_header": "Content criteria",
        "formal_criteria_header": "Formal criteria",
        "title_input_label": "Abstract title",
        "title_placeholder": "Enter the title of your paper...",
        "text_area_label": "Paste the abstract body",
        "text_area_placeholder": "Paste the body of your abstract here (without the title)...",
        "clear_button": "Clear",
        "length_caption": "Body: {aktualna} / {limit} {jednostka}",
        "check_button": "Check abstract",
        "fill_both_caption": "Enter both a title and body to enable the check.",
        "analyzing": "Analyzing the abstract against the criteria for: {kategoria}...",
        "api_error": "Anthropic API error: {e}",
        "unexpected_error": "Unexpected connection error: {e}",
        "parse_warning": "Could not parse the structured response — raw output:",
        "download_button": "Download review (.md)",
        "history_expander": "This session's history ({n})",
        "dim_punktacja_formalna": "Formal",
        "dim_punktacja_merytoryczna": "Content",
        "dim_punktacja_jezykowa": "Language",
        "no_score": "No score",
        "diagnostic_caption": "Diagnostic, informal scores — not a decision by the review committee.",
        "length_ok": "Body length: {aktualna}/{limit} {jednostka} — within limit",
        "length_over": "⚠️ Body length: {aktualna}/{limit} {jednostka} — over the limit",
        "strengths_header": "What works well",
        "improve_header": "What to improve",
        "suggestion_label": "Suggestion:",
        "status_ready": "Ready to submit",
        "status_minor": "Needs minor revisions",
        "status_needs_work": "Needs more work",
        "average_caption": "**{etykieta}** (average of 3 dimensions: {srednia}/100)",
    },
}


def t(key, **kwargs):
    lang = st.session_state.get("ui_lang", "Polski")
    template = TRANSLATIONS.get(lang, TRANSLATIONS["Polski"]).get(key, key)
    return template.format(**kwargs) if kwargs else template


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


def get_access_code():
    code = os.getenv("APP_ACCESS_CODE")
    if not code:
        try:
            code = st.secrets["APP_ACCESS_CODE"]
        except (FileNotFoundError, KeyError):
            code = None
    return code


def wyloguj():
    st.session_state.authenticated = False


def policz_dlugosc(tekst, jednostka_klucz):
    if jednostka_klucz == "slowa":
        return len(tekst.split())
    return len(tekst)


def opis_struktury(kat_cfg):
    if kat_cfg["struktura"] == "sekcje":
        return t("structure_sections", sekcje=", ".join(kat_cfg["sekcje"]))
    return t("structure_continuous")


def kolor_wyniku(punktacja):
    if punktacja >= 80:
        return "#15803D"  # zielony
    if punktacja >= 50:
        return "#B45309"  # bursztynowy
    return "#B91C1C"  # czerwony


def etykieta_wyniku(punktacja):
    if punktacja >= 80:
        return "success", t("status_ready")
    if punktacja >= 50:
        return "warning", t("status_minor")
    return "error", t("status_needs_work")


def gauge_svg(punktacja, size=140):
    punktacja = max(0, min(100, punktacja))
    r = size * 0.4
    stroke = size * 0.09
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    offset = circumference * (1 - punktacja / 100)
    kolor = kolor_wyniku(punktacja)
    font_num = size * 0.24
    return f"""
<div style="display:flex; justify-content:center; margin: 2px 0 4px 0;">
  <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
    <circle cx="{cx}" cy="{cy}" r="{r}" stroke="#E5E7EB" stroke-width="{stroke}" fill="none" />
    <circle cx="{cx}" cy="{cy}" r="{r}" stroke="{kolor}" stroke-width="{stroke}" fill="none"
      stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
      stroke-linecap="round" transform="rotate(-90 {cx} {cy})" />
    <text x="{cx}" y="{cy + font_num * 0.32:.1f}" text-anchor="middle" font-size="{font_num:.0f}"
      font-weight="700" fill="{kolor}" font-family="sans-serif">{punktacja}</text>
  </svg>
</div>
"""


def bezpieczna_punktacja(wynik, klucz):
    try:
        return max(0, min(100, int(wynik.get(klucz))))
    except (TypeError, ValueError):
        return None


def build_system_prompt(kategoria, kat_cfg, limit, jednostka_etykieta, jezyk_abstraktu, ui_lang):
    merytoryczne_txt = "\n".join(f"- {k}" for k in kat_cfg["kryteria_merytoryczne"])
    formalne_txt = "\n".join(f"- {k}" for k in kat_cfg["kryteria_formalne"])
    struktura_txt = opis_struktury(kat_cfg)

    if kat_cfg["struktura"] == "sekcje":
        struktura_instrukcja = (
            f"Ten typ zgłoszenia wymaga wyraźnie wyodrębnionych sekcji: "
            f"{', '.join(kat_cfg['sekcje'])}. Sprawdź, czy abstrakt je zawiera — "
            f"jako nagłówki albo przynajmniej wyraźnie rozróżnialne, kolejne akapity."
        )
    else:
        struktura_instrukcja = (
            "Ten typ zgłoszenia powinien być napisany jako tekst ciągły, bez "
            "wymaganych podnagłówków sekcji — nie sugeruj dodawania nagłówków."
        )

    if ui_lang == "English":
        jezyk_instrukcja = (
            "Write ALL text fields in the JSON response in English, regardless of "
            "the language the abstract itself is written in."
        )
    else:
        jezyk_instrukcja = (
            "Wszystkie teksty w odpowiedzi JSON napisz po polsku, niezależnie od "
            "języka abstraktu."
        )

    return f"""Jesteś asystentem pomagającym studentom medycyny przygotować abstrakt \
(razem z tytułem) przed wysłaniem go do komitetu recenzenckiego. Oceny, łącznie \
z punktacją, to narzędzie diagnostyczne dla studenta — NIE jest to ocena komitetu \
recenzenckiego ani gwarancja przyjęcia na konferencję. Twoim celem jest pomóc \
studentowi poprawić tekst przed formalną recenzją.

Dostajesz tytuł i treść abstraktu jako jedną wiadomość, oznaczone nagłówkami TYTUŁ: \
i TREŚĆ ABSTRAKTU:.

Abstrakt jest napisany w języku: {jezyk_abstraktu}.
Kategoria zgłoszenia: {kategoria}.
Wymagana struktura: {struktura_txt}.
Limit długości treści abstraktu (bez tytułu): {limit} {jednostka_etykieta}.

Nie licz i nie podawaj długości tekstu — ta wartość jest liczona precyzyjnie \
po stronie aplikacji, nie przez Ciebie.

Oceniasz abstrakt w TRZECH NIEZALEŻNYCH wymiarach, każdy w skali 0-100:

## 1. PUNKTACJA FORMALNA ("punktacja_formalna")
Dotyczy formy, nie treści merytorycznej. Weź pod uwagę:
{formalne_txt}
Dodatkowo sprawdź strukturę: {struktura_instrukcja}
Weź pod uwagę też zgodność z limitem długości (podanym wyżej) i to, czy tekst nie \
urywa się w połowie zdania.
Skala: 90-100 pełna zgodność formalna; 70-89 drobne uchybienia formalne; \
40-69 istotne braki formalne (np. brak wymaganej sekcji, przekroczony limit); \
0-39 poważne naruszenia formalne.

## 2. PUNKTACJA MERYTORYCZNA ("punktacja_merytoryczna")
Dotyczy wartości naukowej/klinicznej treści, niezależnie od formy. Weź pod uwagę:
{merytoryczne_txt}
Skala: 90-100 mocne, dobrze skonstruowane badanie/przypadek/przegląd, wnioski \
poparte danymi; 70-89 solidne, ale z drobnymi lukami merytorycznymi; \
40-69 istotne braki merytoryczne (np. brak konkretnych wyników, słabo uzasadnione \
wnioski); 0-39 fundamentalne problemy z wartością merytoryczną.

## 3. PUNKTACJA JĘZYKOWA ("punktacja_jezykowa")
Dotyczy wyłącznie jakości języka, niezależnie od formy i treści. Weź pod uwagę:
- Jasność i precyzja sformułowań
- Brak nieuzasadnionego żargonu bez wyjaśnienia
- Spójność logiczna zdań i akapitów
- Poprawność gramatyczna i stylistyczna (szczególnie istotna, jeśli abstrakt jest \
po angielsku — sprawdź naturalność sformułowań, nie tylko gramatykę)
Skala: 90-100 język jasny, poprawny, dobrze się czyta; 70-89 drobne potknięcia \
językowe niezakłócające zrozumienia; 40-69 język utrudnia zrozumienie w kilku \
miejscach; 0-39 poważne problemy z jasnością lub poprawnością języka.

{jezyk_instrukcja}

Odpowiedz WYŁĄCZNIE poprawnym obiektem JSON, bez żadnego tekstu przed ani po, bez \
znaczników markdown, w dokładnie takiej strukturze:

{{
  "punktacja_formalna": <liczba całkowita 0-100>,
  "punktacja_merytoryczna": <liczba całkowita 0-100>,
  "punktacja_jezykowa": <liczba całkowita 0-100>,
  "mocne_strony": ["punkt 1", "punkt 2"],
  "do_poprawy": [
    {{"fragment": "krótki cytat z tytułu lub treści", "sugestia": "konkretna propozycja zmiany"}}
  ]
}}
"""


def call_claude(client, tytul, tekst, kategoria, kat_cfg, limit, jednostka_etykieta, jezyk_abstraktu, ui_lang):
    system_prompt = build_system_prompt(kategoria, kat_cfg, limit, jednostka_etykieta, jezyk_abstraktu, ui_lang)
    tresc_wiadomosci = f"TYTUŁ:\n{tytul}\n\nTREŚĆ ABSTRAKTU:\n{tekst}"
    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        # Sonnet 5 ma domyślnie włączone adaptacyjne myślenie (bloki "thinking" przed
        # tekstem) i odrzuca niedomyślne parametry samplingu (temperature/top_p/top_k).
        # Ten task to prosta, ustrukturyzowana ocena — wyłączamy myślenie: taniej,
        # szybciej, a odpowiedź zawiera tylko blok tekstowy.
        thinking={"type": "disabled"},
        system=system_prompt,
        messages=[{"role": "user", "content": tresc_wiadomosci}],
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


def render_wynik(wynik, aktualna_dlugosc, limit, jednostka_etykieta):
    punktacje = {}
    kolumny = st.columns(3)
    for kol, klucz in zip(kolumny, WYMIARY):
        wartosc = bezpieczna_punktacja(wynik, klucz)
        punktacje[klucz] = wartosc
        with kol:
            st.markdown(
                f"<p style='text-align:center; font-weight:600; margin-bottom:0;'>{t(f'dim_{klucz}')}</p>",
                unsafe_allow_html=True,
            )
            if wartosc is not None:
                st.markdown(gauge_svg(wartosc), unsafe_allow_html=True)
            else:
                st.caption(t("no_score"))

    dostepne = [v for v in punktacje.values() if v is not None]
    if dostepne:
        srednia = round(sum(dostepne) / len(dostepne))
        status, etykieta = etykieta_wyniku(srednia)
        getattr(st, status)(t("average_caption", etykieta=etykieta, srednia=srednia))
    st.caption(t("diagnostic_caption"))

    if aktualna_dlugosc > limit:
        st.caption(t("length_over", aktualna=aktualna_dlugosc, limit=limit, jednostka=jednostka_etykieta))
    else:
        st.caption(t("length_ok", aktualna=aktualna_dlugosc, limit=limit, jednostka=jednostka_etykieta))

    if wynik.get("mocne_strony"):
        st.markdown(f"**{t('strengths_header')}**")
        for punkt in wynik["mocne_strony"]:
            st.markdown(f"- {punkt}")

    if wynik.get("do_poprawy"):
        st.markdown(f"**{t('improve_header')}**")
        for i, item in enumerate(wynik["do_poprawy"], 1):
            skrot = item.get("fragment", "")[:60]
            with st.expander(f"{i}. {skrot}..."):
                st.markdown(f"> {item.get('fragment', '')}")
                st.markdown(f"**{t('suggestion_label')}** {item.get('sugestia', '')}")


def wynik_do_markdown(wynik, tytul, kategoria, aktualna_dlugosc, limit, jednostka_etykieta):
    linie_wynikow = []
    dostepne = []
    for klucz in WYMIARY:
        wartosc = bezpieczna_punktacja(wynik, klucz)
        nazwa = t(f"dim_{klucz}")
        if wartosc is not None:
            dostepne.append(wartosc)
            linie_wynikow.append(f"- {nazwa}: {wartosc} / 100")
        else:
            linie_wynikow.append(f"- {nazwa}: {t('no_score')}")

    if dostepne:
        srednia = round(sum(dostepne) / len(dostepne))
        etykieta = etykieta_wyniku(srednia)[1]
        podsumowanie = f"{srednia} / 100 ({etykieta})"
    else:
        podsumowanie = t("no_score")

    lines = [
        f"# {kategoria}",
        f"**{t('title_input_label')}:** {tytul}",
        f"**{'Podsumowanie' if st.session_state.get('ui_lang', 'Polski') == 'Polski' else 'Summary'}:** {podsumowanie}",
        "",
        f"## {t('dim_punktacja_formalna')} / {t('dim_punktacja_merytoryczna')} / {t('dim_punktacja_jezykowa')}",
    ]
    lines += linie_wynikow
    lines += ["", f"## {t('strengths_header')}"]
    lines += [f"- {p}" for p in wynik.get("mocne_strony", [])]
    lines += ["", f"## {t('improve_header')}"]
    for item in wynik.get("do_poprawy", []):
        lines.append(f"- **Fragment:** {item.get('fragment', '')}")
        lines.append(f"  **{t('suggestion_label')}** {item.get('sugestia', '')}")
    if aktualna_dlugosc <= limit:
        dl_line = t("length_ok", aktualna=aktualna_dlugosc, limit=limit, jednostka=jednostka_etykieta)
    else:
        dl_line = t("length_over", aktualna=aktualna_dlugosc, limit=limit, jednostka=jednostka_etykieta)
    lines += ["", f"## {dl_line}"]
    return "\n".join(lines)


def clear_abstrakt():
    st.session_state.abstrakt_tekst = ""
    st.session_state.abstrakt_tytul = ""


def render_header():
    logo_col, lang_col = st.columns([3, 1])
    with logo_col:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=260)
    with lang_col:
        st.selectbox(
            "🌐", ["Polski", "English"], key="ui_lang", label_visibility="collapsed",
        )
    st.title(t("app_title"))
    st.caption(t("app_caption"))


# ---------------- UI ----------------

st.set_page_config(page_title="Sprawdź swój abstrakt / Check your abstract", page_icon="📝", layout="centered")

config = load_config()

ACCESS_CODE = get_access_code()

if ACCESS_CODE:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        render_header()
        st.divider()
        wpisany_kod = st.text_input(t("access_code_label"), type="password", key="wpisany_kod")
        if st.button(t("enter_button"), type="primary"):
            if wpisany_kod == ACCESS_CODE:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error(t("wrong_code"))
        st.stop()

client = get_client()

render_header()

if client is None:
    st.error(t("no_api_key"))
    st.stop()

if "historia" not in st.session_state:
    st.session_state.historia = []
if "abstrakt_tekst" not in st.session_state:
    st.session_state.abstrakt_tekst = ""
if "abstrakt_tytul" not in st.session_state:
    st.session_state.abstrakt_tytul = ""

with st.sidebar:
    st.header(t("settings_header"))

    if ACCESS_CODE:
        st.button(t("logout"), on_click=wyloguj, use_container_width=True)
        st.divider()

    kategorie = list(config["kategorie"].keys())
    kategoria = st.selectbox(t("category_label"), kategorie)
    kat_cfg = config["kategorie"][kategoria]

    jednostka_klucz = st.radio(
        t("unit_label"), options=["znaki", "slowa"],
        format_func=lambda k: t(f"unit_{k}"), horizontal=True,
        key="jednostka_klucz",
    )
    ust = JEDNOSTKI[jednostka_klucz]
    jednostka_etykieta = t(f"unit_{jednostka_klucz}_sufiks")
    limit = st.number_input(
        t("limit_label", jednostka=jednostka_etykieta),
        min_value=ust["min"], max_value=ust["max"],
        value=ust["domyslny_limit"], step=ust["krok"],
    )

    jezyk_abstraktu = st.radio(
        t("abstract_lang_label"), options=["Polski", "English"],
        format_func=lambda v: t("lang_polish") if v == "Polski" else t("lang_english"),
        horizontal=True, key="jezyk_abstraktu",
    )

    with st.expander(t("criteria_expander")):
        st.markdown(f"**{t('structure_prefix')}:** {opis_struktury(kat_cfg)}")
        st.markdown(f"**{t('merit_criteria_header')}**")
        for k in kat_cfg["kryteria_merytoryczne"]:
            st.markdown(f"- {k}")
        st.markdown(f"**{t('formal_criteria_header')}**")
        for k in kat_cfg["kryteria_formalne"]:
            st.markdown(f"- {k}")

tytul = st.text_input(
    t("title_input_label"), key="abstrakt_tytul", placeholder=t("title_placeholder"),
)

col_a, col_b = st.columns([4, 1])
with col_a:
    tekst = st.text_area(
        t("text_area_label"), height=280,
        key="abstrakt_tekst", placeholder=t("text_area_placeholder"),
    )
with col_b:
    st.write("")
    st.write("")
    st.button(t("clear_button"), use_container_width=True, on_click=clear_abstrakt)

aktualna_dlugosc = policz_dlugosc(tekst, jednostka_klucz)
procent = min(aktualna_dlugosc / limit, 1.0) if limit else 0.0
st.progress(procent)
kolor = "red" if aktualna_dlugosc > limit else "gray"
st.caption(f":{kolor}[{t('length_caption', aktualna=aktualna_dlugosc, limit=limit, jednostka=jednostka_etykieta)}]")

gotowe_do_sprawdzenia = bool(tytul.strip()) and bool(tekst.strip())

if st.button(t("check_button"), type="primary", disabled=not gotowe_do_sprawdzenia):
    raw = None
    ui_lang = st.session_state.get("ui_lang", "Polski")
    with st.spinner(t("analyzing", kategoria=kategoria)):
        try:
            raw = call_claude(
                client, tytul, tekst, kategoria, kat_cfg, limit, jednostka_etykieta,
                jezyk_abstraktu, ui_lang,
            )
        except APIError as e:
            st.error(t("api_error", e=e))
        except Exception as e:
            st.error(t("unexpected_error", e=e))

    if raw:
        st.divider()
        wynik = None
        try:
            wynik = parse_response(raw)
        except (ValueError, json.JSONDecodeError):
            st.warning(t("parse_warning"))
            st.markdown(raw)

        if wynik:
            with st.container(border=True):
                render_wynik(wynik, aktualna_dlugosc, limit, jednostka_etykieta)
            raport = wynik_do_markdown(
                wynik, tytul, kategoria, aktualna_dlugosc, limit, jednostka_etykieta
            )
            st.download_button(
                t("download_button"), raport,
                file_name=f"ocena_abstraktu_{datetime.now():%Y%m%d_%H%M}.md",
            )
            punktacja_f = bezpieczna_punktacja(wynik, "punktacja_formalna")
            punktacja_m = bezpieczna_punktacja(wynik, "punktacja_merytoryczna")
            punktacja_j = bezpieczna_punktacja(wynik, "punktacja_jezykowa")
            st.session_state.historia.append({
                "Godzina": datetime.now().strftime("%H:%M:%S"),
                t("title_input_label"): tytul[:40] + ("..." if len(tytul) > 40 else ""),
                t("category_label"): kategoria,
                t("dim_punktacja_formalna"): punktacja_f if punktacja_f is not None else "—",
                t("dim_punktacja_merytoryczna"): punktacja_m if punktacja_m is not None else "—",
                t("dim_punktacja_jezykowa"): punktacja_j if punktacja_j is not None else "—",
            })
elif not gotowe_do_sprawdzenia:
    st.caption(t("fill_both_caption"))

if st.session_state.historia:
    st.divider()
    with st.expander(t("history_expander", n=len(st.session_state.historia))):
        st.dataframe(st.session_state.historia, use_container_width=True, hide_index=True)

st.divider()
st.caption("Studenckie Koło Naukowe Innowacji Medycznych · Gdański Uniwersytet Medyczny")
