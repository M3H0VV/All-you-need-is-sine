import pytest
import numpy as np
import os
from src.visualizer import Visualizer
from src.audio_processor import load_audio, STFTAnalyzer


class TestVisualizer:
    """Test class for Visualizer module."""

    @pytest.fixture
    def sample_audio_path(self):
        """Fixture providing path to sample audio file."""
        return os.path.join(os.path.dirname(__file__), '..', 'samples', 'techno_bass.wav')

    @pytest.fixture
    def synthetic_audio(self):
        """Fixture providing synthetic audio signal."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # Create simple sine wave at 440 Hz
        audio = np.sin(2 * np.pi * 440 * t)
        return audio, sr

    @pytest.fixture
    def visualizer(self):
        """Fixture providing Visualizer instance."""
        return Visualizer(sr=22050)

    def test_visualizer_init(self):
        """Test Visualizer initialization."""
        viz = Visualizer(sr=22050)
        assert viz.sr == 22050

    def test_plot_waveform(self, visualizer, synthetic_audio):
        """Test waveform plot generation."""
        audio, sr = synthetic_audio
        fig = visualizer.plot_waveform(audio, title="Test Waveform")

        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Test Waveform"

    def test_plot_magnitude_spectrogram(self, visualizer, sample_audio_path):
        """Test magnitude spectrogram plot generation."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)
        spec = analyzer.get_magnitude_spectrogram()

        fig = visualizer.plot_magnitude_spectrogram(
            spec, sr, hop_length=512, 
            title="Test Magnitude Spectrogram"
        )

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_phase_spectrogram(self, visualizer, sample_audio_path):
        """Test phase spectrogram plot generation."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)
        phase_spec = analyzer.phase_spec

        fig = visualizer.plot_phase_spectrogram(
            phase_spec, sr, hop_length=512,
            title="Test Phase Spectrogram"
        )

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_comparison(self, visualizer, synthetic_audio):
        """Test comparison plot generation."""
        audio, sr = synthetic_audio
        reconstructed = audio * 0.8  # Simulate reduced reconstruction
        
        fig = visualizer.plot_comparison(
            audio, reconstructed,
            title="Test Comparison"
        )

        assert fig is not None
        assert len(fig.data) == 2  # Original and reconstructed

    def test_plot_sinusoids_overlay_empty(self, visualizer, synthetic_audio):
        """Test sinusoids overlay with empty sinusoids list."""
        audio, sr = synthetic_audio
        fig = visualizer.plot_sinusoids_overlay(
            audio, [], num_sinusoids=10,
            title="Test Empty Overlay"
        )

        assert fig is not None
        assert len(fig.data) == 1  # Only original signal

    def test_plot_sinusoids_overlay_with_sinusoids(self, visualizer, sample_audio_path):
        """Test sinusoids overlay with extracted sinusoids."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)
        sinusoids = analyzer.extract_peaks(num_sinusoids=20)

        fig = visualizer.plot_sinusoids_overlay(
            audio, sinusoids, num_sinusoids=10,
            title="Test Overlay with Sinusoids"
        )

        assert fig is not None
        assert len(fig.data) == 2  # Original and reconstructed

    def test_plot_frequency_peaks_empty(self, visualizer):
        """Test frequency peaks plot with empty list."""
        fig = visualizer.plot_frequency_peaks([], top_n=20)

        assert fig is not None

    def test_plot_frequency_peaks_with_data(self, visualizer, sample_audio_path):
        """Test frequency peaks plot with extracted sinusoids."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)
        sinusoids = analyzer.extract_peaks(num_sinusoids=30)

        fig = visualizer.plot_frequency_peaks(sinusoids, top_n=15)

        assert fig is not None
        assert len(fig.data) > 0
