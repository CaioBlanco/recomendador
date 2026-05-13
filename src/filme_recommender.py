"""
filme_recommender.py
--------------------
Sistema de recomendação de filmes baseado em similaridade de conteúdo.

A ideia é simples: se você gostou de um filme, provavelmente vai gostar
de outros que têm gêneros, descrições e características parecidas.
Isso se chama "filtragem baseada em conteúdo" (content-based filtering).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FilmeRecommender:
    def __init__(self):
        # Aqui vamos guardar o dataframe e a matriz de similaridade
        # depois que o modelo for treinado
        self.df = None
        self.matriz_similaridade = None

    def carregar_dados(self, caminho: str) -> pd.DataFrame:
        """
        Carrega o dataset de filmes do CSV.
        Esperamos colunas como: title, genres, overview, vote_average.
        """
        df = pd.read_csv(caminho)

        # Vamos usar só as colunas que nos interessam
        colunas_necessarias = ["title", "genres", "overview", "vote_average"]
        df = df[colunas_necessarias].dropna()

        # Remove duplicatas pelo título
        df = df.drop_duplicates(subset="title")
        df = df.reset_index(drop=True)

        print(f"✅ {len(df)} filmes carregados!")
        return df

    def treinar(self, df: pd.DataFrame):
        """
        Treina o modelo de recomendação.

        O truque aqui é criar uma "sopa de texto" para cada filme,
        juntando gênero e descrição. Depois usamos TF-IDF para
        transformar esse texto em números e calculamos o quão
        parecido cada filme é com os outros.
        """
        self.df = df.copy()

        # Cria uma coluna combinando gênero e descrição do filme
        # Repetimos o gênero para dar mais peso a ele
        self.df["sopa"] = (
            self.df["genres"].fillna("") + " " +
            self.df["genres"].fillna("") + " " +
            self.df["overview"].fillna("")
        )

        # TF-IDF transforma o texto em vetores numéricos
        # "english" remove palavras sem significado como "the", "a", "is"
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        matriz_tfidf = tfidf.fit_transform(self.df["sopa"])

        # Similaridade de cosseno: mede o ângulo entre dois vetores
        # 1.0 = idênticos, 0.0 = completamente diferentes
        self.matriz_similaridade = cosine_similarity(matriz_tfidf, matriz_tfidf)

        print(f"✅ Modelo treinado com {len(self.df)} filmes!")

    def recomendar(self, titulo: str, n: int = 5) -> pd.DataFrame:
        """
        Recebe o título de um filme e retorna os N mais similares.

        Exemplo:
            recomendar("The Dark Knight", n=5)
            → retorna os 5 filmes mais parecidos
        """
        if self.df is None:
            raise ValueError("Treine o modelo primeiro com .treinar()")

        # Busca o filme pelo título (ignora maiúsculas/minúsculas)
        titulos = self.df["title"].str.lower()
        titulo_lower = titulo.lower()

        if titulo_lower not in titulos.values:
            # Tenta busca parcial se não encontrar exato
            matches = self.df[titulos.str.contains(titulo_lower, na=False)]
            if matches.empty:
                return pd.DataFrame()
            idx = matches.index[0]
        else:
            idx = titulos[titulos == titulo_lower].index[0]

        # Pega as similaridades desse filme com todos os outros
        scores = list(enumerate(self.matriz_similaridade[idx]))

        # Ordena do mais similar para o menos (ignora o próprio filme)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]  # pula o índice 0 (o próprio filme)

        # Monta o dataframe de resultado
        indices = [i[0] for i in scores]
        resultado = self.df.iloc[indices][["title", "genres", "vote_average"]].copy()
        resultado["similaridade"] = [round(i[1], 3) for i in scores]
        resultado = resultado.rename(columns={
            "title": "Título",
            "genres": "Gêneros",
            "vote_average": "Nota",
            "similaridade": "Similaridade"
        })

        return resultado.reset_index(drop=True)

    def buscar_titulos(self, query: str, n: int = 10) -> list:
        """
        Retorna títulos que contêm o texto digitado.
        Usado para o autocomplete na interface.
        """
        if self.df is None:
            return []

        matches = self.df[
            self.df["title"].str.lower().str.contains(query.lower(), na=False)
        ]["title"].head(n).tolist()

        return matches
