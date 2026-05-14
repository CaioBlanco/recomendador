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

# Adiciona a pasta raiz do projeto ao path do Python.
# Sem isso, o import "from src.filme_recommender import ..." falharia
# dependendo de onde o script fosse chamado.
sys.path.append(os.path.dirname(__file__))
from src.filme_recommender import FilmeRecommender
from src.musica_recommender import MusicaRecommender


# ── Configuração da página ─────────────────────────────────
# DEVE ser a primeira chamada Streamlit do script.
# Define o título que aparece na aba do navegador, o ícone e o layout.
# layout="wide" faz o app usar toda a largura da tela em vez de uma coluna central.
st.set_page_config(
    page_title="Recomendador de Filmes e Músicas",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬🎵 Sistema de Recomendação")
st.markdown("Descubra filmes e músicas parecidos com o que você já ama!")
st.divider()  # Linha horizontal separadora


# ── Cache: carrega os modelos só uma vez ───────────────────
# O Streamlit re-executa o script inteiro a cada interação do usuário.
# Sem @st.cache_resource, o modelo seria retreinado a cada clique — o que
# levaria vários segundos. Com o cache, o modelo é carregado uma vez e
# reutilizado em todas as interações seguintes.

@st.cache_resource
def carregar_modelo_filmes():
    """Carrega e treina o recomendador de filmes."""
    caminho = "data/movies.csv"

    # Retorna None se o dataset não existir — a interface trata esse caso
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

    # Retorna None se o dataset não existir — a interface trata esse caso
    if not os.path.exists(caminho):
        return None

    rec = MusicaRecommender()
    df = rec.carregar_dados(caminho)
    rec.treinar(df)
    return rec


# ── Carrega os modelos ─────────────────────────────────────
# st.spinner exibe uma animação de carregamento enquanto os modelos são treinados.
# Isso só demora na primeira execução — depois o cache cuida do resto.
with st.spinner("Carregando modelos... isso pode levar alguns segundos na primeira vez!"):
    rec_filmes = carregar_modelo_filmes()
    rec_musicas = carregar_modelo_musicas()


# ── Abas: Filmes e Músicas ─────────────────────────────────
# st.tabs cria abas navegáveis na interface.
# Cada aba é um contexto separado: o código dentro de "with aba_filmes"
# só renderiza elementos na aba de filmes, e vice-versa.
aba_filmes, aba_musicas = st.tabs(["🎬 Filmes", "🎵 Músicas"])


# ════════════════════════════════════════════════
# ABA DE FILMES
# ════════════════════════════════════════════════
with aba_filmes:
    st.header("Recomendação de Filmes")
    st.markdown("Digite o nome de um filme que você gostou e descubra filmes parecidos!")

    if rec_filmes is None:
        # Dataset não encontrado: exibe instruções de como baixar
        st.warning("⚠️ Dataset de filmes não encontrado!")
        st.markdown("""
        **Como baixar:**
        1. Acesse [TMDB Movies Dataset no Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
        2. Baixe o arquivo `tmdb_5000_movies.csv`
        3. Renomeie para `movies.csv` e coloque na pasta `data/`
        4. Reinicie o app com `streamlit run app.py`
        """)
    else:
        # st.columns divide a linha em colunas proporcionais.
        # [3, 1] significa: coluna 1 ocupa 75% da largura, coluna 2 ocupa 25%.
        col1, col2 = st.columns([3, 1])

        with col1:
            # Campo de texto para o usuário digitar o nome do filme
            titulo = st.text_input(
                "🔍 Nome do filme",
                placeholder="Ex: The Dark Knight, Inception, Interstellar..."
            )

        with col2:
            # Slider para escolher quantas recomendações exibir (mín: 3, máx: 10, padrão: 5)
            n_filmes = st.slider("Quantas recomendações?", 3, 10, 5)

        if titulo:
            # Busca filmes cujo título contém o texto digitado (autocomplete)
            sugestoes = rec_filmes.buscar_titulos(titulo, n=8)

            if sugestoes:
                # Exibe um dropdown com os filmes encontrados para o usuário escolher.
                # options=list() é necessário para compatibilidade com Python 3.14+
                filme_selecionado = st.selectbox("Selecione o filme:", options=list(sugestoes))

                # Botão principal que dispara a recomendação
                if st.button("🎬 Recomendar filmes similares", type="primary"):
                    with st.spinner("Buscando filmes parecidos..."):
                        resultado = rec_filmes.recomendar(filme_selecionado, n=n_filmes)

                    if resultado is None or resultado.empty:
                        # Filme não encontrado na matriz de similaridade
                        st.error("Filme não encontrado. Tente outro título.")
                    else:
                        # Exibe a tabela de recomendações
                        st.success(f"Filmes parecidos com **{filme_selecionado}**:")
                        st.dataframe(
                            resultado,
                            use_container_width=True,  # ocupa toda a largura disponível
                            hide_index=True            # esconde o índice numérico do DataFrame
                        )
            else:
                # Nenhum filme encontrado com o texto digitado
                st.info("Nenhum filme encontrado com esse nome. Tente outro!")


# ════════════════════════════════════════════════
# ABA DE MÚSICAS
# ════════════════════════════════════════════════
with aba_musicas:
    st.header("Recomendação de Músicas")
    st.markdown("Digite o nome de uma música e descubra outras com som parecido!")

    if rec_musicas is None:
        # Dataset não encontrado: exibe instruções de como baixar
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
            # Campo de texto para o usuário digitar o nome da música
            nome_musica = st.text_input(
                "🔍 Nome da música",
                placeholder="Ex: Blinding Lights, Shape of You, Bohemian Rhapsody..."
            )

        with col2:
            # Slider para escolher quantas recomendações exibir (mín: 3, máx: 10, padrão: 5)
            n_musicas = st.slider("Quantas recomendações? ", 3, 10, 5)

        if nome_musica:
            # Busca músicas cujo nome contém o texto digitado
            # Retorna strings no formato "Música — Artista"
            sugestoes = rec_musicas.buscar_musicas(nome_musica, n=8)

            if sugestoes:
                musica_selecionada = st.selectbox("Selecione a música:", sugestoes)

                # Extrai só o nome da música (sem o artista) para passar ao recomendador.
                # O formato retornado por buscar_musicas é "Música — Artista",
                # então separamos pelo " — " e pegamos a primeira parte.
                nome_sel = musica_selecionada.split(" — ")[0]

                # Botão principal que dispara a recomendação
                if st.button("🎵 Recomendar músicas similares", type="primary"):
                    with st.spinner("Analisando características de áudio..."):
                        resultado = rec_musicas.recomendar(nome_sel, n=n_musicas)

                    if resultado is None or resultado.empty:
                        # Música não encontrada na matriz de similaridade
                        st.error("Música não encontrada. Tente outro título.")
                    else:
                        # Exibe a tabela de recomendações
                        st.success(f"Músicas parecidas com **{musica_selecionada}**:")
                        st.dataframe(
                            resultado,
                            use_container_width=True,  # ocupa toda a largura disponível
                            hide_index=True            # esconde o índice numérico do DataFrame
                        )
            else:
                # Nenhuma música encontrada com o texto digitado
                st.info("Nenhuma música encontrada com esse nome. Tente outra!")


# ── Rodapé ────────────────────────────────────────────────
st.divider()
# unsafe_allow_html=True permite renderizar o HTML dos links diretamente
st.markdown(
    "Feito por **Caio Blanco** · "
    "[GitHub](https://github.com/CaioBlanco) · "
    "[LinkedIn](https://www.linkedin.com/in/caio-blanco-501267253/)",
    unsafe_allow_html=True
)