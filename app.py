"""
app.py
------
Interface web do sistema de recomendação, feita com Streamlit.

Streamlit é incrível para isso: você escreve Python puro e ele
vira uma página web bonita automaticamente. Sem HTML, sem CSS, sem JavaScript.

Para rodar:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from src.filme_recommender import FilmeRecommender
from src.musica_recommender import MusicaRecommender


# ── Configuração da página ─────────────────────────────────
st.set_page_config(
    page_title="Recomendador de Filmes e Músicas",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬🎵 Sistema de Recomendação")
st.markdown("Descubra filmes e músicas parecidos com o que você já ama!")
st.divider()


# ── Cache: carrega os modelos só uma vez ───────────────────
# O @st.cache_resource faz o modelo ser carregado apenas na primeira vez
# Sem isso, o modelo seria retreinado a cada interação do usuário!

@st.cache_resource
def carregar_modelo_filmes():
    """Carrega e treina o recomendador de filmes."""
    caminho = "data/movies.csv"
    if not os.path.exists(caminho):
        return None

    rec = FilmeRecommender()
    df = rec.carregar_dados(caminho)
    rec.treinar(df)
    return rec


@st.cache_resource
def carregar_modelo_musicas():
    """Carrega e treina o recomendador de músicas."""
    caminho = "data/spotify_tracks.csv"
    if not os.path.exists(caminho):
        return None

    rec = MusicaRecommender()
    df = rec.carregar_dados(caminho)
    rec.treinar(df)
    return rec


# ── Carrega os modelos ─────────────────────────────────────
with st.spinner("Carregando modelos... isso pode levar alguns segundos na primeira vez!"):
    rec_filmes = carregar_modelo_filmes()
    rec_musicas = carregar_modelo_musicas()


# ── Abas: Filmes e Músicas ─────────────────────────────────
aba_filmes, aba_musicas = st.tabs(["🎬 Filmes", "🎵 Músicas"])


# ════════════════════════════════════════════════
# ABA DE FILMES
# ════════════════════════════════════════════════
with aba_filmes:
    st.header("Recomendação de Filmes")
    st.markdown("Digite o nome de um filme que você gostou e descubra filmes parecidos!")

    if rec_filmes is None:
        # Aviso amigável caso o arquivo não exista
        st.warning("⚠️ Dataset de filmes não encontrado!")
        st.markdown("""
        **Como baixar:**
        1. Acesse [TMDB Movies Dataset no Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
        2. Baixe o arquivo `tmdb_5000_movies.csv`
        3. Renomeie para `movies.csv` e coloque na pasta `data/`
        4. Reinicie o app com `streamlit run app.py`
        """)
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            titulo = st.text_input(
                "🔍 Nome do filme",
                placeholder="Ex: The Dark Knight, Inception, Interstellar..."
            )

        with col2:
            n_filmes = st.slider("Quantas recomendações?", 3, 10, 5)

        if titulo:
            # Busca filmes que contêm o texto digitado
            sugestoes = rec_filmes.buscar_titulos(titulo, n=8)

            if sugestoes:
                filme_selecionado = st.selectbox("Selecione o filme:", sugestoes)

                if st.button("🎬 Recomendar filmes similares", type="primary"):
                    with st.spinner("Buscando filmes parecidos..."):
                        resultado = rec_filmes.recomendar(filme_selecionado, n=n_filmes)

                    if resultado.empty:
                        st.error("Filme não encontrado. Tente outro título.")
                    else:
                        st.success(f"Filmes parecidos com **{filme_selecionado}**:")
                        st.dataframe(
                            resultado,
                            use_container_width=True,
                            hide_index=True
                        )
            else:
                st.info("Nenhum filme encontrado com esse nome. Tente outro!")


# ════════════════════════════════════════════════
# ABA DE MÚSICAS
# ════════════════════════════════════════════════
with aba_musicas:
    st.header("Recomendação de Músicas")
    st.markdown("Digite o nome de uma música e descubra outras com som parecido!")

    if rec_musicas is None:
        st.warning("⚠️ Dataset de músicas não encontrado!")
        st.markdown("""
        **Como baixar:**
        1. Acesse [Spotify Tracks Dataset no Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
        2. Baixe o arquivo `dataset.csv`
        3. Renomeie para `spotify_tracks.csv` e coloque na pasta `data/`
        4. Reinicie o app com `streamlit run app.py`
        """)
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            nome_musica = st.text_input(
                "🔍 Nome da música",
                placeholder="Ex: Blinding Lights, Shape of You, Bohemian Rhapsody..."
            )

        with col2:
            n_musicas = st.slider("Quantas recomendações? ", 3, 10, 5)

        if nome_musica:
            sugestoes = rec_musicas.buscar_musicas(nome_musica, n=8)

            if sugestoes:
                musica_selecionada = st.selectbox("Selecione a música:", sugestoes)
                nome_sel = musica_selecionada.split(" — ")[0]

                if st.button("🎵 Recomendar músicas similares", type="primary"):
                    with st.spinner("Analisando características de áudio..."):
                        resultado = rec_musicas.recomendar(nome_sel, n=n_musicas)

                    if resultado.empty:
                        st.error("Música não encontrada. Tente outro título.")
                    else:
                        st.success(f"Músicas parecidas com **{musica_selecionada}**:")
                        st.dataframe(
                            resultado,
                            use_container_width=True,
                            hide_index=True
                        )
            else:
                st.info("Nenhuma música encontrada com esse nome. Tente outra!")


# ── Rodapé ────────────────────────────────────────────────
st.divider()
st.markdown(
    "Feito por **Caio Blanco** · "
    "[GitHub](https://github.com/CaioBlanco) · "
    "[LinkedIn](https://www.linkedin.com/in/caio-blanco-501267253/)",
    unsafe_allow_html=True
)
