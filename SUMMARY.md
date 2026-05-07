# Podsumowanie projektu: All You Need Is Sine

## O projekcie
Projekt **"All You Need Is Sine"** to aplikacja webowa napisana w języku Python z użyciem biblioteki Streamlit oraz pakiet do przetwarzania dźwięku. Jej głównym celem jest dekompozycja dowolnego sygnału dźwiękowego na komponenty sinusoidalne przy użyciu Krótkoczasowej Transformaty Fouriera (STFT) oraz demonstracja, w jaki sposób z tych sinusoid można z powrotem zrekonstruować oryginalny dźwięk.

## Architektura i struktura plików

Kod został podzielony na odseparowane i łatwe w testowaniu moduły, zgrupowane w folderze `src/`.

### 1. Interfejs Użytkownika (`app.py`)
- Skrypt główny pełniący funkcję interfejsu graficznego (Streamlit).
- Pozwala użytkownikom przesyłać pliki audio (WAV, MP3) i wyświetla interaktywne suwaki do konfiguracji STFT (n_fft, hop_length) oraz wyboru liczby sinusoid do rekonstrukcji.
- Integruje wszystkie moduły systemu w celu wykonania analizy, zarysowania wykresów, odtworzenia dźwięku w przeglądarce i udostępnienia go do pobrania.

### 2. Przetwarzanie Dźwięku (`src/audio_processor.py`)
- **Narzędzia audio**: Funkcje `load_audio` i `get_audio_info` (wykorzystujące bibliotekę *librosa*) obsługujące wczytywanie plików do sygnału jednokanałowego (mono).
- **Klasa `STFTAnalyzer`**: Serce analizy. Wylicza macierz STFT, spektrogramy fazowe i amplitudowe (w skali dB) oraz wykorzystuje funkcję `scipy.signal.find_peaks` do ekstrakcji najbardziej prominentnych pików częstotliwościowych z każdej ramki czasowej.

### 3. Syntezator (`src/synthesizer.py`)
- **Klasa `SinusoidalSynthesizer`**: Odpowiada za proces odwrotny do analizy – generuje sygnał na podstawie fal sinusoidalnych.
- Oferuje różne metody odbudowy dźwięku, w tym `reconstruct_from_stft_peaks` dla rekonstrukcji stricte na podstawie pików oraz `reconstruct_from_istft`, która modyfikuje i odwraca macierz STFT w celu otrzymania wygładzonego sygnału.
- Posiada metody umożliwiające badanie jakości rekonstrukcji (MSE, RMSE, współczynnik SNR w dB).

### 4. Wizualizacja (`src/visualizer.py`)
- **Klasa `Visualizer`**: Warstwa prezentacyjna oparta o bibliotekę *Plotly*.
- Generuje wysoce interaktywne i estetyczne wizualizacje przebiegu fali dźwiękowej (waveform), spektrogramów amplitudowych i fazowych, wykresów słupkowych reprezentujących poszczególne piki częstotliwościowe oraz porównań sygnału oryginalnego z zrekonstruowanym.

### 5. Testy Jednostkowe (`tests/`)
- Zespół testów jednostkowych napisanych z użyciem frameworka **pytest**.
- **Fixtury** przygotowują próbki zarówno syntetyczne (idealna fala o częstotliwości 440 Hz), jak i korzystają z przykładowych krótkich sampli rzeczywistych z dysku.
- Sprawdzane jest tam poprawne inicjowanie modułów, poprawność matematyczna STFT, parametry wykresów, a także poziomy błędów odtwarzania.

## Wykorzystane Technologie
- **Język:** Python 3
- **Interfejs UI:** Streamlit
- **Przetwarzanie Sygnałów / DSP:** Librosa, SciPy
- **Operacje Numeryczne:** NumPy
- **Wizualizacja:** Plotly
- **Obsługa Plików Audio:** Soundfile
- **Testowanie:** Pytest

Projekt zachowuje wzorową czystość, korzysta z typowania zmiennych (type hints) oraz przejrzystych docstringów.