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

JEDNOSTKI = {
    "Znaki": {"etykieta": "znaków", "domyslny_limit": 2000, "min": 200, "max": 10000, "krok": 100},
    "Słowa": {"etykieta": "słów", "domyslny_limit": 300, "min": 30, "max": 2000, "krok": 10},
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


def policz_dlugosc(tekst, jednostka):
    if jednostka == "Słowa":
        return len(tekst.split())
    return len(tekst)


def opis_struktury(kat_cfg):
    if kat_cfg["struktura"] == "sekcje":
        return "wymagane wyodrębnione sekcje: " + ", ".join(kat_cfg["sekcje"])
    return "tekst ciągły, bez wymaganych podnagłówków sekcji"


def kolor_wyniku(punktacja):
    if punktacja >= 80:
        return "#0F6E5E"  # zielony, zgodny z motywem aplikacji
    if punktacja >= 50:
        return "#D97706"  # bursztynowy
    return "#DC2626"  # czerwony


def etykieta_wyniku(punktacja):
    if punktacja >= 80:
        return "success", "Gotowy do wysłania"
    if punktacja >= 50:
        return "warning", "Wymaga drobnych poprawek"
    return "error", "Wymaga jeszcze pracy"


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


def bezpieczna_punktacja(wynik, klucz="punktacja"):
    try:
        return max(0, min(100, int(wynik.get(klucz))))
    except (TypeError, ValueError):
        return None


WYMIARY = [
    ("punktacja_formalna", "Formalna"),
    ("punktacja_merytoryczna", "Merytoryczna"),
    ("punktacja_jezykowa", "Językowa"),
]


def build_system_prompt(kategoria, kat_cfg, limit, jednostka_etykieta, jezyk):
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

    return f"""Jesteś asystentem pomagającym studentom medycyny przygotować abstrakt \
(razem z tytułem) przed wysłaniem go do komitetu recenzenckiego. Oceny, łącznie \
z punktacją, to narzędzie diagnostyczne dla studenta — NIE jest to ocena komitetu \
recenzenckiego ani gwarancja przyjęcia na konferencję. Twoim celem jest pomóc \
studentowi poprawić tekst przed formalną recenzją.

Dostajesz tytuł i treść abstraktu jako jedną wiadomość, oznaczone nagłówkami TYTUŁ: \
i TREŚĆ ABSTRAKTU:.

Abstrakt jest napisany w języku: {jezyk}.
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

Odpowiedz WYŁĄCZNIE poprawnym obiektem JSON, bez żadnego tekstu przed ani po, bez \
znaczników markdown, w dokładnie takiej strukturze (wszystkie teksty po polsku, \
niezależnie od języka abstraktu):

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


def call_claude(client, tytul, tekst, kategoria, kat_cfg, limit, jednostka_etykieta, jezyk):
    system_prompt = build_system_prompt(kategoria, kat_cfg, limit, jednostka_etykieta, jezyk)
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
    for kol, (klucz, nazwa) in zip(kolumny, WYMIARY):
        wartosc = bezpieczna_punktacja(wynik, klucz)
        punktacje[klucz] = wartosc
        with kol:
            st.markdown(
                f"<p style='text-align:center; font-weight:600; margin-bottom:0;'>{nazwa}</p>",
                unsafe_allow_html=True,
            )
            if wartosc is not None:
                st.markdown(gauge_svg(wartosc), unsafe_allow_html=True)
            else:
                st.caption("Brak wyniku")

    dostepne = [v for v in punktacje.values() if v is not None]
    if dostepne:
        srednia = round(sum(dostepne) / len(dostepne))
        status, etykieta = etykieta_wyniku(srednia)
        getattr(st, status)(f"**{etykieta}** (średnia z 3 wymiarów: {srednia}/100)")
    st.caption("Wyniki pomocnicze, diagnostyczne — nie są oceną komitetu recenzenckiego.")

    if aktualna_dlugosc > limit:
        st.caption(f"⚠️ Długość treści: {aktualna_dlugosc}/{limit} {jednostka_etykieta} — przekracza limit")
    else:
        st.caption(f"Długość treści: {aktualna_dlugosc}/{limit} {jednostka_etykieta} — mieści się")

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


def wynik_do_markdown(wynik, tytul, kategoria, aktualna_dlugosc, limit, jednostka_etykieta):
    linie_wynikow = []
    dostepne = []
    for klucz, nazwa in WYMIARY:
        wartosc = bezpieczna_punktacja(wynik, klucz)
        if wartosc is not None:
            dostepne.append(wartosc)
            linie_wynikow.append(f"- {nazwa}: {wartosc} / 100")
        else:
            linie_wynikow.append(f"- {nazwa}: brak wyniku")

    if dostepne:
        srednia = round(sum(dostepne) / len(dostepne))
        etykieta = etykieta_wyniku(srednia)[1]
        podsumowanie = f"Średnia: {srednia} / 100 ({etykieta})"
    else:
        podsumowanie = "brak wyników"

    lines = [
        f"# Ocena abstraktu — {kategoria}",
        f"**Tytuł:** {tytul}",
        f"**Podsumowanie:** {podsumowanie}",
        "",
        "## Wyniki szczegółowe",
    ]
    lines += linie_wynikow
    lines += ["", "## Co działa dobrze"]
    lines += [f"- {p}" for p in wynik.get("mocne_strony", [])]
    lines += ["", "## Do poprawy"]
    for item in wynik.get("do_poprawy", []):
        lines.append(f"- **Fragment:** {item.get('fragment', '')}")
        lines.append(f"  **Sugestia:** {item.get('sugestia', '')}")
    status_dlugosci = "mieści się" if aktualna_dlugosc <= limit else (
        f"przekracza limit o {aktualna_dlugosc - limit} {jednostka_etykieta}"
    )
    lines += ["", f"## Długość treści: {aktualna_dlugosc} / {limit} {jednostka_etykieta} ({status_dlugosci})"]
    return "\n".join(lines)


def clear_abstrakt():
    st.session_state.abstrakt_tekst = ""
    st.session_state.abstrakt_tytul = ""


# ---------------- UI ----------------

st.set_page_config(page_title="Sprawdź swój abstrakt", page_icon="📝", layout="centered")

config = load_config()
ACCESS_CODE = get_access_code()

if ACCESS_CODE:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("📝 Sprawdź swój abstrakt")
        st.caption("Wstępna, nieformalna ocena przed wysłaniem do komitetu recenzenckiego")
        st.divider()
        wpisany_kod = st.text_input("Kod dostępu", type="password", key="wpisany_kod")
        if st.button("Wejdź", type="primary"):
            if wpisany_kod == ACCESS_CODE:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Nieprawidłowy kod dostępu.")
        st.stop()

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
if "abstrakt_tytul" not in st.session_state:
    st.session_state.abstrakt_tytul = ""

with st.sidebar:
    st.header("Ustawienia")

    if ACCESS_CODE:
        st.button("Wyloguj", on_click=wyloguj, use_container_width=True)
        st.divider()

    kategorie = list(config["kategorie"].keys())
    kategoria = st.selectbox("Kategoria zgłoszenia", kategorie)
    kat_cfg = config["kategorie"][kategoria]

    jednostka = st.radio("Licz limit w:", list(JEDNOSTKI.keys()), horizontal=True)
    ust = JEDNOSTKI[jednostka]
    limit = st.number_input(
        f"Limit treści ({ust['etykieta']})", min_value=ust["min"], max_value=ust["max"],
        value=ust["domyslny_limit"], step=ust["krok"],
    )

    jezyk = st.radio("Język abstraktu", ["Polski", "English"], horizontal=True)

    with st.expander("Kryteria oceny dla tej kategorii"):
        st.markdown(f"**Struktura:** {opis_struktury(kat_cfg)}")
        st.markdown("**Kryteria merytoryczne**")
        for k in kat_cfg["kryteria_merytoryczne"]:
            st.markdown(f"- {k}")
        st.markdown("**Kryteria formalne**")
        for k in kat_cfg["kryteria_formalne"]:
            st.markdown(f"- {k}")

tytul = st.text_input(
    "Tytuł abstraktu", key="abstrakt_tytul", placeholder="Wpisz tytuł pracy...",
)

col_a, col_b = st.columns([4, 1])
with col_a:
    tekst = st.text_area(
        "Wklej treść abstraktu", height=280,
        key="abstrakt_tekst", placeholder="Wklej tutaj treść swojego abstraktu (bez tytułu)...",
    )
with col_b:
    st.write("")
    st.write("")
    st.button("Wyczyść", use_container_width=True, on_click=clear_abstrakt)

aktualna_dlugosc = policz_dlugosc(tekst, jednostka)
procent = min(aktualna_dlugosc / limit, 1.0) if limit else 0.0
st.progress(procent)
kolor = "red" if aktualna_dlugosc > limit else "gray"
st.caption(f":{kolor}[Treść: {aktualna_dlugosc} / {limit} {ust['etykieta']}]")

gotowe_do_sprawdzenia = bool(tytul.strip()) and bool(tekst.strip())

if st.button("Sprawdź abstrakt", type="primary", disabled=not gotowe_do_sprawdzenia):
    raw = None
    with st.spinner(f"Analizuję abstrakt pod kątem kryteriów: {kategoria}..."):
        try:
            raw = call_claude(client, tytul, tekst, kategoria, kat_cfg, limit, ust["etykieta"], jezyk)
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
            render_wynik(wynik, aktualna_dlugosc, limit, ust["etykieta"])
            raport = wynik_do_markdown(
                wynik, tytul, kategoria, aktualna_dlugosc, limit, ust["etykieta"]
            )
            st.download_button(
                "Pobierz ocenę (.md)", raport,
                file_name=f"ocena_abstraktu_{datetime.now():%Y%m%d_%H%M}.md",
            )
            punktacja_f = bezpieczna_punktacja(wynik, "punktacja_formalna")
            punktacja_m = bezpieczna_punktacja(wynik, "punktacja_merytoryczna")
            punktacja_j = bezpieczna_punktacja(wynik, "punktacja_jezykowa")
            st.session_state.historia.append({
                "Godzina": datetime.now().strftime("%H:%M:%S"),
                "Tytuł": tytul[:40] + ("..." if len(tytul) > 40 else ""),
                "Kategoria": kategoria,
                "Formalna": punktacja_f if punktacja_f is not None else "—",
                "Merytoryczna": punktacja_m if punktacja_m is not None else "—",
                "Językowa": punktacja_j if punktacja_j is not None else "—",
                "Długość": f"{aktualna_dlugosc}/{limit} {ust['etykieta']}",
            })
elif not gotowe_do_sprawdzenia:
    st.caption("Wpisz tytuł i treść abstraktu, żeby odblokować sprawdzenie.")

if st.session_state.historia:
    st.divider()
    with st.expander(f"Historia tej sesji ({len(st.session_state.historia)})"):
        st.dataframe(st.session_state.historia, use_container_width=True, hide_index=True)
