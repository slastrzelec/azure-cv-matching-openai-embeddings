"""
PDF Handler - Moduł do ekstrakcji tekstu z plików PDF
Obsługuje dwie biblioteki: pdfplumber (zalecane) i PyPDF2
"""

import os
import PyPDF2
import pdfplumber


def extract_text_from_pdf(pdf_path, method='pdfplumber', clean=True):
    """
    Wyciąga tekst z pliku PDF z obsługą błędów.
    
    Args:
        pdf_path (str): Ścieżka do pliku PDF
        method (str): 'pdfplumber' lub 'pypdf2' (domyślnie: 'pdfplumber')
        clean (bool): Czy czyścić tekst - usuwa nadmiarowe spacje/newliny (domyślnie: True)
    
    Returns:
        str: Wyekstraktowany tekst lub None w przypadku błędu
        
    Example:
        >>> text = extract_text_from_pdf('cv.pdf', method='pdfplumber')
        >>> print(len(text))
        9668
    """
    
    try:
        # Sprawdź czy plik istnieje
        if not os.path.exists(pdf_path):
            print(f"❌ Błąd: Plik nie istnieje: {pdf_path}")
            return None
        
        # Ekstrakcja tekstem z pdfplumber
        if method == 'pdfplumber':
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Sprawdź czy strona ma tekst
                        text += page_text + "\n"
        
        # Ekstrakcja tekstem z PyPDF2
        elif method == 'pypdf2':
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        else:
            print(f"❌ Błąd: Nieznana metoda '{method}'. Użyj 'pdfplumber' lub 'pypdf2'")
            return None
        
        # Czyszczenie tekstu (opcjonalne)
        if clean and text:
            # Usuń nadmiarowe spacje i newliny
            text = ' '.join(text.split())
        
        return text
    
    except Exception as e:
        print(f"❌ Błąd podczas ekstrakcji: {e}")
        return None


def compare_pdf_methods(pdf_path):
    """
    Porównuje PyPDF2 i pdfplumber dla danego PDF.
    Testuje obie metody i zwraca statystyki z rekomendacją.
    
    Args:
        pdf_path (str): Ścieżka do pliku PDF
    
    Returns:
        dict: Słownik z wynikami:
            {
                'pdfplumber': {'length': int, 'text': str},
                'pypdf2': {'length': int, 'text': str},
                'recommended': str  # 'pdfplumber', 'pypdf2' lub 'both'
            }
        lub None w przypadku błędu
        
    Example:
        >>> results = compare_pdf_methods('cv.pdf')
        >>> print(results['recommended'])
        'pdfplumber'
    """
    
    print("🔍 PORÓWNANIE METOD EKSTRAKCJI")
    print("=" * 60)
    
    # Test pdfplumber
    text_pdfplumber = extract_text_from_pdf(pdf_path, method='pdfplumber', clean=True)
    
    # Test PyPDF2
    text_pypdf2 = extract_text_from_pdf(pdf_path, method='pypdf2', clean=True)
    
    # Sprawdź czy ekstrakcja się udała
    if not text_pdfplumber or not text_pypdf2:
        print("❌ Nie udało się wyekstraktować tekstu obiema metodami")
        return None
    
    # Przygotuj wyniki
    results = {
        'pdfplumber': {
            'length': len(text_pdfplumber),
            'text': text_pdfplumber
        },
        'pypdf2': {
            'length': len(text_pypdf2),
            'text': text_pypdf2
        }
    }
    
    # Wyświetl porównanie
    print(f"\n📊 STATYSTYKI:")
    print(f"{'Metoda':<15} {'Długość':<10} {'Ocena'}")
    print("-" * 60)
    print(f"{'pdfplumber':<15} {results['pdfplumber']['length']:<10} {'✅ Lepsze formatowanie'}")
    print(f"{'PyPDF2':<15} {results['pypdf2']['length']:<10} {'⚠️ Prostsze, szybsze'}")
    
    # Określ rekomendację
    diff = abs(results['pdfplumber']['length'] - results['pypdf2']['length'])
    
    print(f"\n💡 REKOMENDACJA:")
    if diff < 100:
        print("   Obie metody dają podobne wyniki - wybierz dowolną")
        results['recommended'] = 'both'
    else:
        print("   pdfplumber - lepsze formatowanie dla złożonych PDF")
        results['recommended'] = 'pdfplumber'
    
    return results


# Funkcja pomocnicza do testowania modułu
if __name__ == "__main__":
    print("🧪 Moduł pdf_handler.py")
    print("Zaimportuj funkcje w innych plikach:")
    print("  from utils.pdf_handler import extract_text_from_pdf, compare_pdf_methods")