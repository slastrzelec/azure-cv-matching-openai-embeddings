"""
Embeddings Module - Moduł do generowania embeddingów i matchingu CV z ofertami pracy
Wykorzystuje OpenAI API (text-embedding-3-small/large)
"""

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os


def get_openai_client(api_key=None):
    """
    Tworzy klienta OpenAI.
    
    Args:
        api_key (str, optional): Klucz API OpenAI. Jeśli None, pobiera z os.getenv()
    
    Returns:
        OpenAI: Klient OpenAI lub None przy błędzie
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Błąd: Brak klucza OPENAI_API_KEY")
        return None
    
    return OpenAI(api_key=api_key)


def calculate_embedding(text, model="text-embedding-3-small", client=None):
    """
    Generuje embedding dla podanego tekstu.
    
    Args:
        text (str): Tekst do przekształcenia w embedding
        model (str): Model OpenAI ('text-embedding-3-small' lub 'text-embedding-3-large')
        client (OpenAI, optional): Klient OpenAI. Jeśli None, tworzy nowy
    
    Returns:
        np.array: Wektor embeddingu lub None przy błędzie
        
    Example:
        >>> embedding = calculate_embedding("Python developer")
        >>> print(len(embedding))
        1536
    """
    try:
        # Utwórz klienta jeśli nie podano
        if client is None:
            client = get_openai_client()
            if client is None:
                return None
        
        # Wywołaj API
        result = client.embeddings.create(
            input=[text],
            model=model,
        )
        
        # Zamień na numpy array
        embedding = np.array(result.data[0].embedding)
        
        return embedding
    
    except Exception as e:
        print(f"❌ Błąd podczas generowania embeddingu: {e}")
        return None


def extract_skills_with_ai(cv_text, model="gpt-4o-mini", client=None):
    """
    Ekstraktuje kluczowe umiejętności z CV używając OpenAI.
    
    Args:
        cv_text (str): Pełny tekst CV
        model (str): Model OpenAI do ekstrakcji (domyślnie: gpt-4o-mini)
        client (OpenAI, optional): Klient OpenAI
    
    Returns:
        str: Lista umiejętności oddzielonych przecinkami lub None przy błędzie
        
    Example:
        >>> skills = extract_skills_with_ai(cv_text)
        >>> print(skills)
        'Python, PyTorch, scikit-learn, time series, MLflow, Docker, AWS'
    """
    try:
        # Utwórz klienta jeśli nie podano
        if client is None:
            client = get_openai_client()
            if client is None:
                return None
        
        # Prompt dla ekstrakcji
        extraction_prompt = f"""
Analyze the CV below and extract ONLY the most important technical skills.

Response format: comma-separated list of skills, no additional explanations.

Example: Python, PyTorch, scikit-learn, time series forecasting, MLflow, Docker, AWS SageMaker

CV:
{cv_text}

Skills:
"""
        
        # Wywołaj API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=300,
            temperature=0  # Deterministyczna odpowiedź
        )
        
        # Wyciągnij wynik
        extracted_skills = response.choices[0].message.content.strip()
        
        return extracted_skills
    
    except Exception as e:
        print(f"❌ Błąd podczas ekstrakcji skills: {e}")
        return None


def calculate_job_similarity(cv_embedding, job_embeddings):
    """
    Oblicza cosine similarity między CV a listą ofert pracy.
    
    Args:
        cv_embedding (np.array): Embedding CV
        job_embeddings (list): Lista embeddingów ofert pracy
    
    Returns:
        list: Lista podobieństw (float 0-1) dla każdej oferty
        
    Example:
        >>> similarities = calculate_job_similarity(cv_emb, [job1_emb, job2_emb])
        >>> print(similarities)
        [0.6137, 0.4976]
    """
    similarities = []
    
    for job_embedding in job_embeddings:
        similarity = cosine_similarity([cv_embedding], [job_embedding])[0][0]
        similarities.append(similarity)
    
    return similarities


def rank_jobs(jobs, similarities, top_n=5):
    """
    Rankuje oferty pracy według podobieństwa i zwraca top N.
    
    Args:
        jobs (list): Lista słowników z ofertami pracy
        similarities (list): Lista wartości podobieństwa (float)
        top_n (int): Ile najlepszych ofert zwrócić (domyślnie: 5)
    
    Returns:
        list: Posortowana lista ofert z dodanym polem 'similarity'
        
    Example:
        >>> ranked = rank_jobs(jobs, [0.61, 0.49, 0.45], top_n=3)
        >>> print(ranked[0]['title'])
        'Senior Machine Learning Engineer'
    """
    # Dodaj podobieństwo do każdej oferty
    for job, similarity in zip(jobs, similarities):
        job['similarity'] = similarity
    
    # Sortuj od najwyższego podobieństwa
    ranked_jobs = sorted(jobs, key=lambda x: x['similarity'], reverse=True)
    
    # Zwróć top N
    return ranked_jobs[:top_n]


def get_similarity_rating(similarity):
    """
    Zwraca ocenę tekstową dla wartości podobieństwa.
    
    Args:
        similarity (float): Wartość cosine similarity (0-1)
    
    Returns:
        tuple: (emoji, rating_text, color)
        
    Example:
        >>> emoji, rating, color = get_similarity_rating(0.65)
        >>> print(f"{emoji} {rating}")
        '🟢 Excellent match'
    """
    similarity_pct = similarity * 100
    
    if similarity_pct > 60:
        return "🟢", "Excellent match", "green"
    elif similarity_pct > 50:
        return "🟠", "Good match", "orange"
    elif similarity_pct > 40:
        return "🟡", "Average match", "yellow"
    else:
        return "🔴", "Poor match", "red"


# Funkcja pomocnicza do testowania modułu
if __name__ == "__main__":
    print("🧪 Moduł embeddings.py")
    print("Zaimportuj funkcje w innych plikach:")
    print("  from utils.embeddings import calculate_embedding, extract_skills_with_ai")
    print("  from utils.embeddings import calculate_job_similarity, rank_jobs")
