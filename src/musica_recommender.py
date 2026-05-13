"""
musica_recommender.py
---------------------
Sistema de recomendação de músicas baseado em características de áudio.

Diferente dos filmes (onde usamos texto), aqui usamos dados numéricos
como energia, dançabilidade, valência (se a música é alegre ou triste),
entre outros. Isso é bem mais próximo do que o Spotify faz de verdade!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Essas são as características de áudio que o Spotify fornece para cada música
# Vamos usar elas para calcular a similaridade entre músicas
FEATURES_AUDIO = [
    "danceability",   # quão dançável é a música (0 a 1)
    "energy",         # intensidade e atividade (0 a 1)
    "valence",        # positividade musical (0 = triste, 1 = alegre)
    "tempo",          # batidas por minuto (BPM)
    "acousticness",   # quão acústica é (0 a 1)
    "instrumentalness", # se tem vocal ou não (0 a 1)
    "liveness",       # se parece ao vivo (0 a 1)
    "speechiness",    # quantidade de palavras faladas (0 a 1)
]


class MusicaRecommender:
    def __init__(self):
        self.df = None
        self.matriz_similaridade = None
        self.scaler = StandardScaler()  # vai normalizar os dados numéricos

    def carregar_dados(self, caminho: str) -> pd.DataFrame:
        """
        Carrega o dataset de músicas do CSV.
        Esperamos colunas como: track_name, artists, track_genre e as features de áudio.
        """
        df = pd.read_csv(caminho)

        # Colunas que precisamos
        colunas_texto = ["track_name", "artists", "track_genre"]
        colunas_necessarias = colunas_texto + FEATURES_AUDIO

        # Filtra só as colunas que existem no dataset
        colunas_existentes = [c for c in colunas_necessarias if c in df.columns]
        df = df[colunas_existentes].dropna()

        # Remove duplicatas pelo nome da música + artista
        df = df.drop_duplicates(subset=["track_name", "artists"])
        df = df.reset_index(drop=True)

        print(f"✅ {len(df)} músicas carregadas!")
        return df

    def treinar(self, df: pd.DataFrame):
        """
        Treina o modelo de recomendação de músicas.

        Normalizamos as features numéricas (para que o tempo em BPM
        não domine sobre características de 0 a 1) e calculamos
        a similaridade entre todas as músicas.
        """
        self.df = df.copy()

        # Pega só as features que existem no dataset
        features_disponiveis = [f for f in FEATURES_AUDIO if f in self.df.columns]

        if not features_disponiveis:
            raise ValueError("Nenhuma feature de áudio encontrada no dataset!")

        # Normaliza: coloca todas as features na mesma escala
        # Sem isso, o BPM (que vai de 60 a 200) dominaria tudo
        X = self.scaler.fit_transform(self.df[features_disponiveis])

        # Calcula a similaridade entre todas as músicas
        self.matriz_similaridade = cosine_similarity(X, X)

        print(f"✅ Modelo treinado com {len(self.df)} músicas!")

    def recomendar(self, nome_musica: str, artista: str = None, n: int = 5) -> pd.DataFrame:
        """
        Recebe o nome de uma música e retorna as N mais similares.

        Exemplo:
            recomendar("Shape of You", artista="Ed Sheeran", n=5)
            → retorna as 5 músicas com características de áudio mais parecidas
        """
        if self.df is None:
            raise ValueError("Treine o modelo primeiro com .treinar()")

        # Busca a música
        nomes = self.df["track_name"].str.lower()
        nome_lower = nome_musica.lower()

        if artista:
            # Busca por nome + artista se fornecido
            mask = (
                self.df["track_name"].str.lower().str.contains(nome_lower, na=False) &
                self.df["artists"].str.lower().str.contains(artista.lower(), na=False)
            )
        else:
            mask = self.df["track_name"].str.lower().str.contains(nome_lower, na=False)

        matches = self.df[mask]

        if matches.empty:
            return pd.DataFrame()

        idx = matches.index[0]

        # Calcula e ordena as similaridades
        scores = list(enumerate(self.matriz_similaridade[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]

        # Monta o resultado
        indices = [i[0] for i in scores]
        resultado = self.df.iloc[indices][["track_name", "artists", "track_genre"]].copy()
        resultado["similaridade"] = [round(i[1], 3) for i in scores]
        resultado = resultado.rename(columns={
            "track_name": "Música",
            "artists": "Artista",
            "track_genre": "Gênero",
            "similaridade": "Similaridade"
        })

        return resultado.reset_index(drop=True)

    def buscar_musicas(self, query: str, n: int = 10) -> list:
        """
        Retorna músicas que contêm o texto digitado.
        Usado para o autocomplete na interface.
        """
        if self.df is None:
            return []

        matches = self.df[
            self.df["track_name"].str.lower().str.contains(query.lower(), na=False)
        ][["track_name", "artists"]].head(n)

        # Retorna "Música - Artista" para facilitar a seleção
        return [f"{row['track_name']} — {row['artists']}" for _, row in matches.iterrows()]
