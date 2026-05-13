# 🎬🎵 Sistema de Recomendação de Filmes e Músicas

Sistema de recomendação com interface web interativa, desenvolvido em Python com **Streamlit**. Recomenda filmes com base em similaridade de conteúdo e músicas com base em características de áudio — bem parecido com o que o Spotify e a Netflix fazem!

## 🖥️ Interface

A aplicação roda no navegador e possui duas abas:
- **🎬 Filmes** — recomendação por gênero e descrição
- **🎵 Músicas** — recomendação por características de áudio (energia, dançabilidade, valência...)

## 📁 Estrutura

```
recomendador/
│
├── src/
│   ├── filme_recommender.py    # Modelo de recomendação de filmes
│   └── musica_recommender.py   # Modelo de recomendação de músicas
│
├── data/                       # Datasets (baixar do Kaggle)
│   ├── movies.csv
│   └── spotify_tracks.csv
│
├── app.py                      # Interface web (Streamlit)
└── requirements.txt
```

## 🛠️ Tecnologias

| Biblioteca      | Uso                                      |
|-----------------|------------------------------------------|
| `scikit-learn`  | TF-IDF, similaridade de cosseno, scaler |
| `pandas`        | Manipulação dos datasets                 |
| `streamlit`     | Interface web interativa                 |

## 🤖 Como Funciona

### Filmes — Filtragem por Conteúdo
1. Combina gênero + descrição de cada filme em um texto
2. Usa **TF-IDF** para transformar o texto em vetores numéricos
3. Calcula a **similaridade de cosseno** entre os vetores
4. Retorna os filmes com maior similaridade

### Músicas — Filtragem por Características de Áudio
1. Usa features do Spotify: `danceability`, `energy`, `valence`, `tempo`...
2. Normaliza os dados com **StandardScaler**
3. Calcula a **similaridade de cosseno** entre as músicas
4. Retorna as músicas com perfil de áudio mais parecido

## 🚀 Como Rodar

### 1. Clone o repositório
```bash
git clone https://github.com/CaioBlanco/recomendador.git
cd recomendador
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe os datasets no Kaggle
- **Filmes:** [TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) → renomeie para `movies.csv` → coloque em `data/`
- **Músicas:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) → renomeie para `spotify_tracks.csv` → coloque em `data/`

### 4. Rode o app
```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador 🚀

## 👤 Autor

Feito por **Caio Blanco Schaidhauer** —
[LinkedIn](https://www.linkedin.com/in/caio-blanco-501267253/) ·
[GitHub](https://github.com/CaioBlanco)
