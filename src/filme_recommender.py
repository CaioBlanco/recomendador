"""
filme_recommender.py
--------------------
Sistema de recomendação de filmes baseado em similaridade de conteúdo.

A ideia central é simples: representamos cada filme como um "texto"
combinando título, gêneros e sinopse, depois calculamos o quão parecidos
esses textos são entre si usando TF-IDF + similaridade de cosseno.
"""

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FilmeRecommender:
    def __init__(self):
        # O DataFrame com os dados dos filmes, preenchido em carregar_dados()
        self.df = None

        # Matriz N×N onde cada célula [i][j] é o score de similaridade
        # entre o filme i e o filme j (valores de 0 a 1)
        self.matriz_similaridade = None

    def carregar_dados(self, caminho: str) -> pd.DataFrame:
        """
        Lê o CSV, seleciona apenas as colunas necessárias e limpa os dados.

        O dataset do TMDB armazena os gêneros como uma string JSON
        (ex: "[{'id': 28, 'name': 'Action'}]"), então precisamos
        converter isso para um texto legível como "Action, Drama".
        """
        df = pd.read_csv(caminho)

        # Mantemos só o que vamos usar: título, gêneros, sinopse e nota
        colunas_necessarias = ["title", "genres", "overview", "vote_average"]
        df = df[colunas_necessarias].dropna()

        # Remove filmes com o mesmo título para evitar confusão na busca
        df = df.drop_duplicates(subset="title")
        df = df.reset_index(drop=True)

        def extrair_generos(genres_str):
            """
            Converte a string JSON de gêneros em texto separado por vírgula.
            Ex: "[{'id': 28, 'name': 'Action'}]" → "Action"
            Se a conversão falhar (formato diferente), retorna o valor original.
            """
            try:
                genres = ast.literal_eval(genres_str)  # string → lista de dicts
                return ", ".join([g["name"] for g in genres])
            except:
                return genres_str  # fallback: retorna como está

        df["genres"] = df["genres"].apply(extrair_generos)
        print(f"✅ {len(df)} filmes carregados!")
        return df

    def treinar(self, df: pd.DataFrame):
        """
        Constrói a matriz de similaridade entre todos os filmes.

        Passos:
        1. Cria uma "sopa de palavras" para cada filme, unindo título,
           gêneros e sinopse em um único texto.
        2. Aplica TF-IDF para transformar esses textos em vetores numéricos.
           TF-IDF valoriza palavras importantes e ignora as muito comuns.
        3. Calcula a similaridade de cosseno entre todos os pares de filmes.
           O resultado é uma matriz N×N (N = número de filmes).
        """
        self.df = df.copy()

        # "Sopa": texto único por filme que resume suas características.
        # Título aparece primeiro para ter mais peso na representação.
        self.df["sopa"] = (
            self.df["title"].fillna("") + " " +
            self.df["genres"].fillna("") + " " +
            self.df["overview"].fillna("")
        )

        # TF-IDF: transforma textos em vetores numéricos.
        # stop_words="english" remove palavras sem significado como "the", "is".
        # max_features=5000 limita o vocabulário para economizar memória.
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        matriz_tfidf = tfidf.fit_transform(self.df["sopa"])

        # Similaridade de cosseno: mede o ângulo entre dois vetores.
        # Valor 1.0 = filmes idênticos, 0.0 = completamente diferentes.
        # Essa operação é a mais pesada: O(N²) em memória e tempo.
        self.matriz_similaridade = cosine_similarity(matriz_tfidf, matriz_tfidf)
        print(f"✅ Modelo treinado com {len(self.df)} filmes!")

    def recomendar(self, titulo: str, n: int = 5) -> pd.DataFrame:
        """
        Retorna os N filmes mais similares ao título fornecido.

        A busca é case-insensitive. Se o título exato não for encontrado,
        tenta um match parcial (ex: "dark" encontra "The Dark Knight").

        Retorna um DataFrame vazio se nenhum filme for encontrado.
        """
        if self.df is None:
            raise ValueError("Treine o modelo primeiro com .treinar()")

        # Normaliza tudo para minúsculas para comparação sem distinção de maiúsculas
        titulos = self.df["title"].str.lower()
        titulo_lower = titulo.lower()

        if titulo_lower not in titulos.values:
            # Busca parcial: aceita títulos que contenham a query
            matches = self.df[titulos.str.contains(titulo_lower, na=False)]
            if matches.empty:
                return pd.DataFrame()  # Nenhum filme encontrado
            idx = matches.index[0]    # Pega o primeiro match
        else:
            # Match exato encontrado
            idx = titulos[titulos == titulo_lower].index[0]

        # Pega a linha da matriz correspondente ao filme selecionado.
        # Cada valor nessa linha é a similaridade com outro filme.
        scores = list(enumerate(self.matriz_similaridade[idx]))

        # Ordena do mais similar para o menos similar
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Descarta o índice 0, que é o próprio filme (similaridade = 1.0),
        # e pega os N seguintes
        scores = scores[1:n+1]

        # Monta o DataFrame de resultado com os filmes recomendados
        indices = [i[0] for i in scores]
        resultado = self.df.iloc[indices][["title", "genres", "vote_average"]].copy()
        resultado["similaridade"] = [round(i[1], 3) for i in scores]

        # Renomeia colunas para português para exibição no Streamlit
        resultado = resultado.rename(columns={
            "title": "Título",
            "genres": "Gêneros",
            "vote_average": "Nota",
            "similaridade": "Similaridade"
        })
        return resultado.reset_index(drop=True)

    def buscar_titulos(self, query: str, n: int = 10) -> list:
        """
        Retorna uma lista de títulos que contêm o texto digitado.
        Usado para popular o selectbox de autocomplete na interface.
        """
        if self.df is None:
            return []

        # Filtra filmes cujo título contém a query (case-insensitive)
        matches = self.df[
            self.df["title"].str.lower().str.contains(query.lower(), na=False)
        ]["title"].head(n).tolist()

        return matches