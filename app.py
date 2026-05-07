import io
import os

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from src.audio_processor import load_audio, get_audio_info, STFTAnalyzer
from src.synthesizer import SinusoidalSynthesizer
from src.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="All You Need Is Sine",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
    }
    /* Powiększenie tytułów zakładek (tabs) */
    button[data-baseweb="tab"] p, 
    button[data-baseweb="tab"] span {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    /* Rozciągnięcie zakładek na całą szerokość ekranu / kontenera */
    button[data-baseweb="tab"] {
        flex: 1 !important;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def process_uploaded_audio(file_bytes: bytes, file_name: str, max_duration: float):
    """Ładuje i przycina plik audio, zapisując wynik w pamięci podręcznej (cache)."""
    ext = os.path.splitext(file_name)[1].lower()
    if not ext:
        ext = ".wav"
    temp_path = f"temp_audio{ext}"
    
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    audio, sr = load_audio(temp_path)
    audio_info = get_audio_info(audio, sr)
    
    was_trimmed = False
    if audio_info['duration'] > max_duration:
        audio = audio[:int(max_duration * sr)]
        audio_info = get_audio_info(audio, sr)
        was_trimmed = True
        
    orig_bytes = io.BytesIO()
    sf.write(orig_bytes, audio, sr, format='WAV', subtype='PCM_16')
    orig_audio_data = orig_bytes.getvalue()
    
    return audio, sr, audio_info, was_trimmed, orig_audio_data

@st.cache_resource(show_spinner=False)
def compute_stft_cached(audio: np.ndarray, sr: int, n_fft: int, hop_length: int, window: str, center: bool):
    """Oblicza STFT i zapisuje wynik w pamięci podręcznej dla określonych parametrów i sygnału."""
    analyzer = STFTAnalyzer(n_fft=n_fft, hop_length=hop_length, window=window, center=center)
    analyzer.compute_stft(audio, sr)
    spec_db = analyzer.get_magnitude_spectrogram()
    return analyzer, spec_db

# Title
st.markdown('<h1 class="title">🎵 All You Need Is Sine</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Rozkład dowolnego dźwięku na sinusoidy STFT oraz odszumianie widmowe</p>', unsafe_allow_html=True)

st.markdown("""
Każdy dźwięk można przedstawić jako sumę sinusoid o różnych częstotliwościach.
Ta aplikacja demonstruje, jak poprzez dekompozycję dźwięku za pomocą STFT można go
zrekonstruować przy użyciu różnej liczby składowych sinusoidalnych, a także jak
skutecznie usuwać niechciany szum z tła, wykorzystując bramkowanie w dziedzinie częstotliwości.

### Jak to działa:

1. **Wgraj** plik audio (WAV, MP3 lub AMR)
2. **Analizuj** za pomocą STFT, aby wyodrębnić składowe sinusoidalne
3. **Eksploruj**, dostosowując suwaki kontrolujące jakość rekonstrukcji i stopień odszumiania
4. **Posłuchaj**, jak dźwięk buduje się z sinusoid oraz jak bramka szumów wpływa na jego charakter

Miłego odkrywania! 🎵

💡 **Wskazówka:** Zaawansowane ustawienia analizy (Rozmiar okna, Długość skoku) znajdują się w pływającym panelu bocznym. **Kliknij ikonę strzałki (`>`) w lewym górnym rogu ekranu, aby je wysunąć.**
""")

# Sidebar configuration
st.sidebar.header("🎚️ Parametry STFT")
st.sidebar.markdown("Zmień parametry matematyczne analizy. Mniejsze okno to lepsza precyzja w czasie, większe - w częstotliwości.")

n_fft_selection = st.sidebar.select_slider("Rozmiar okna (Window size)", options=[256, 512, 1024, 2048, 4096, 8192, "Wszystkie próbki (FT)"], value=2048)
overlap_selection = st.sidebar.select_slider(
    "Procent nakładania się okien", 
    options=["0%", "50%", "75%", "87.5%"], 
    value="75%", 
    help="Określa, w jakim stopniu okna analizy nakładają się na siebie. Większe nakładanie to płynniejszy obraz spektrogramu, ale wolniejsze obliczenia."
)

# ============================================================================
# Main app
# ============================================================================

# Initialize session state for caching
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None


# --- Audio Source Selection ---
st.header("📁 Wybierz źródło dźwięku")

source_option = st.radio(
    "Wybierz źródło",
    ("Wgraj własny plik", "Wybierz z biblioteki sampli", "Nagraj dźwięk (Mikrofon)"),
    horizontal=True,
    label_visibility="collapsed"
)

# Helper class to mimic st.UploadedFile for sample files
class FileObject:
    def __init__(self, name, data):
        self.name = name
        self._data = data
        self.size = len(data)
    
    def getvalue(self):
        return self._data

uploaded_file = None

if source_option == "Wgraj własny plik":
    uploaded_file = st.file_uploader("Wybierz plik audio", type=["wav", "mp3", "amr"], label_visibility="collapsed")
elif source_option == "Wybierz z biblioteki sampli":
    SAMPLES_DIR = "samples"
    try:
        sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(('.wav', '.mp3'))])
        if not sample_files:
            st.warning("Nie znaleziono plików w bibliotece sampli.")
        else:
            selected_sample = st.selectbox(
                "Wybierz sampla z listy:", 
                sample_files, 
                index=None,
                placeholder="Wybierz plik...",
                label_visibility="collapsed",
                format_func=lambda x: os.path.splitext(x)[0]
            )
            if selected_sample:
                sample_file_path = os.path.join(SAMPLES_DIR, selected_sample)
                with open(sample_file_path, 'rb') as f:
                    file_bytes = f.read()
                uploaded_file = FileObject(name=selected_sample, data=file_bytes)
    except FileNotFoundError:
        st.error(f"Nie znaleziono folderu z samplami: '{SAMPLES_DIR}'")
elif source_option == "Nagraj dźwięk (Mikrofon)":
    uploaded_file = st.audio_input("Nagraj swój głos lub dźwięk", label_visibility="collapsed")

if uploaded_file is not None:
    # 1. Ograniczenie rozmiaru wgranego pliku (np. max 15 MB)
    MAX_FILE_SIZE_MB = 15
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ Plik jest za duży ({uploaded_file.size / (1024*1024):.2f} MB). Maksymalny dopuszczalny rozmiar to {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    # Save to temporary file
    with st.spinner("Ładowanie pliku audio..."):
        MAX_DURATION_SEC = 10.0
        
        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "audio.wav"
        audio, sr, audio_info, was_trimmed, orig_audio_data = process_uploaded_audio(
            uploaded_file.getvalue(), file_name, MAX_DURATION_SEC
        )
        
        if was_trimmed:
            st.warning(f"⚠️ Wgrany utwór jest zbyt długi ({audio_info['duration']:.2f} s). W celu zapewnienia płynności działania aplikacji i rysowania wykresów, dźwięk został ucięty do pierwszych {int(MAX_DURATION_SEC)} sekund.")
        
        # Store in session state
        st.session_state.audio_data = (audio, sr)
        
        # Display audio info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Czas trwania", f"{audio_info['duration']:.2f}s")
        with col2:
            st.metric("Próbkowanie", f"{sr} Hz")
        with col3:
            st.metric("Próbki", f"{audio_info['samples']:,}")
    
    # ====================================================================
    # STFT Analysis
    # ====================================================================
    
    # Resolve STFT parameters based on selections
    is_full_analysis = n_fft_selection == "Wszystkie próbki (FT)"
    n_fft = len(audio) if is_full_analysis else int(n_fft_selection)
    
    if is_full_analysis:
        hop_length = len(audio)
    else:
        if overlap_selection == "0%":
            hop_length = n_fft
        elif overlap_selection == "50%":
            hop_length = n_fft // 2
        elif overlap_selection == "75%":
            hop_length = n_fft // 4
        else:  # "87.5%"
            hop_length = n_fft // 8
            
    window_type = 'boxcar' if is_full_analysis else 'hann'
    center_stft = False if is_full_analysis else True
    
    with st.spinner("Obliczanie STFT..."):
        analyzer, spec_db = compute_stft_cached(audio, sr, n_fft, hop_length, window_type, center_stft)
        st.session_state.analyzer = analyzer
        
    # Create visualizer
    visualizer = Visualizer(sr=sr)
    
    # Display plots sequentially (ideal for mobile layout)
    fig_wave = visualizer.plot_waveform(audio, "Oryginalny przebieg fali")
    st.plotly_chart(fig_wave, use_container_width=True)
    
    if is_full_analysis:
        # For full analysis, show a 1D frequency spectrum plot
        st.info("ℹ️ Wybrano analizę całego sygnału (FT). Wykres przedstawia widmo częstotliwości dla całego nagrania.")
        fig_spec = visualizer.plot_frequency_spectrum(
            magnitude_db_frame=spec_db[:, 0], # Take the single frame
            sr=sr,
            n_fft=n_fft,
            title="Widmo częstotliwości (dB)"
        )
    else:
        # For STFT, show the 2D/3D spectrogram
        dim_main = st.segmented_control("Zmień wymiar", ["2D", "3D"], default="2D", key="main_spec_3d")
        use_3d_main_spec = dim_main == "3D"
        fig_spec = visualizer.plot_magnitude_spectrogram(
            spec_db, sr, hop_length,
            "Spektrogram amplitudowy (dB)",
            use_3d=use_3d_main_spec
        )
    st.plotly_chart(fig_spec, use_container_width=True)
    
    tab_synteza, tab_odszumianie = st.tabs(["🎛️ Synteza", "🧹 Odszumianie"])

    # ====================================================================
    # Sinusoidal Reconstruction
    # ====================================================================
    with tab_synteza:
        
        st.markdown("W tej sekcji możesz zrekonstruować wgrany plik audio na podstawie wyciągniętych składowych. Za pomocą suwaka zdecyduj, ile najważniejszych sinusoid ma zostać użytych. Zwróć uwagę, jak dźwięk nabiera kształtu i traci robotyczny charakter wraz z dodawaniem kolejnych częstotliwości!")
        
        # Slider to control number of sinusoids
        duration = len(audio) / sr
        max_possible = n_fft // 2
        
        fixed_values = [1, 2, 3, 4, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        valid_options = [v for v in fixed_values if v <= max_possible]
        if not valid_options:
            valid_options = [1]
            
        default_val = 100 if is_full_analysis else 25
        if default_val not in valid_options:
            default_val = valid_options[-1]
            
        num_sinusoids_slider = st.select_slider(
            "Liczba sinusoid do rekonstrukcji",
            options=valid_options,
            value=default_val,
            help="Zwiększ, aby uzyskać lepsze przybliżenie oryginalnego dźwięku"
        )
        
        # Reconstruction
        with st.spinner("Rekonstrukcja dźwięku..."):
            synthesizer = SinusoidalSynthesizer(sr=sr)
            reconstructed = synthesizer.reconstruct_from_istft(
                analyzer.stft_matrix,
                hop_length=hop_length,
                num_sinusoids=num_sinusoids_slider,
                window=analyzer.window,
                length=len(audio),
                center=center_stft
            )
            
            # Calculate error metrics
            errors = synthesizer.get_reconstruction_error(audio, reconstructed)
        
        # Display reconstruction comparison
        st.subheader(f"🔄 Rekonstrukcja (przy użyciu {num_sinusoids_slider} składowych)")
        
        st.metric("Jakość rekonstrukcji (SNR)", f"{errors['snr']:.2f} dB")
        
        # Display comparison plot
        fig = visualizer.plot_comparison(audio, reconstructed)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display audio comparison
        st.subheader("🔊 Porównanie dźwięku")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Oryginalny dźwięk**")
            st.audio(orig_audio_data, format="audio/wav")
            
        with col2:
            st.markdown("**Zrekonstruowany dźwięk**")
            # Convert to audio bytes for playback
            max_val = np.max(np.abs(reconstructed))
            if max_val > 1.0:
                reconstructed = reconstructed / max_val
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, reconstructed, sr, format='WAV', subtype='PCM_16')
            audio_bytes.seek(0)
            
            st.audio(audio_bytes, format="audio/wav")

    # ====================================================================
    # Noise Reduction (Odszumianie)
    # ====================================================================
    with tab_odszumianie:
        
        st.markdown("Zastosuj bramkę szumów (Noise Gate) w dziedzinie częstotliwości. Ustawiając odpowiedni próg wyciszasz najcichsze dźwięki (np. szum tła, szum mikrofonu), zachowując tylko głośne i wyraźne składowe sygnału.")
        
        noise_gate_slider = st.slider(
            "Próg odszumiania (Noise Gate) w dB",
            min_value=-80,
            max_value=0,
            value=-40,
            step=1,
            help="Wycisza częstotliwości poniżej określonego progu głośności."
        )
        
        with st.spinner("Odszumianie dźwięku..."):
            # Calculate gated spectrogram for visualization
            stft_copy = analyzer.stft_matrix.copy()
            mag = np.abs(stft_copy)
            if np.max(mag) > 0:
                mag_db = librosa.amplitude_to_db(mag, ref=np.max)
                mask = mag_db < noise_gate_slider
                stft_copy[mask] = 0
            
            gated_spec_db = librosa.power_to_db(np.abs(stft_copy)**2, ref=np.max)
            
            # Reconstruct gated audio
            gated_audio = synthesizer.reconstruct_from_istft(
                analyzer.stft_matrix,
                hop_length=hop_length,
                noise_gate_db=noise_gate_slider,
                keep_noise=False,
                window=analyzer.window,
                length=len(audio),
                center=center_stft
            )
            
            # Reconstruct noise audio
            noise_audio = synthesizer.reconstruct_from_istft(
                analyzer.stft_matrix,
                hop_length=hop_length,
                noise_gate_db=noise_gate_slider,
                keep_noise=True,
                window=analyzer.window,
                length=len(audio),
                center=center_stft
            )
        
        # Plots
        if is_full_analysis:
            fig_gated_spec = visualizer.plot_frequency_spectrum(
                magnitude_db_frame=gated_spec_db[:, 0],
                sr=sr,
                n_fft=n_fft,
                title="Odszumione widmo częstotliwości (dB)"
            )
        else:
            dim_gated = st.segmented_control("Zmień wymiar", ["2D", "3D"], default="2D", key="gated_spec_3d")
            use_3d_gated_spec = dim_gated == "3D"
            fig_gated_spec = visualizer.plot_magnitude_spectrogram(
                gated_spec_db, sr, hop_length,
                "Odszumiony spektrogram amplitudowy (dB)",
                use_3d=use_3d_gated_spec
            )
        st.plotly_chart(fig_gated_spec, use_container_width=True)
        
        fig_triple_comp = visualizer.plot_triple_comparison(audio, gated_audio, noise_audio)
        st.plotly_chart(fig_triple_comp, use_container_width=True)
        
        # Audio players
        st.subheader("🔊 Porównanie dźwięku (Odszumianie)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Oryginalny dźwięk**")
            st.audio(orig_audio_data, format="audio/wav")
        with col2:
            st.markdown("**Odszumiony dźwięk**")
            max_val = np.max(np.abs(gated_audio))
            if max_val > 1.0: 
                gated_audio = gated_audio / max_val
            gated_bytes = io.BytesIO()
            sf.write(gated_bytes, gated_audio, sr, format='WAV', subtype='PCM_16')
            gated_bytes.seek(0)
            st.audio(gated_bytes, format="audio/wav")
        with col3:
            st.markdown("**Wycięty szum**")
            max_val_n = np.max(np.abs(noise_audio))
            if max_val_n > 1.0: 
                noise_audio = noise_audio / max_val_n
            noise_bytes = io.BytesIO()
            sf.write(noise_bytes, noise_audio, sr, format='WAV', subtype='PCM_16')
            noise_bytes.seek(0)
            st.audio(noise_bytes, format="audio/wav")

else:
    st.info("👆 Wgraj plik audio lub wybierz sampla z biblioteki, aby rozpocząć!")
