import pytest
import numpy as np
import os
from src.audio_processor import load_audio, get_audio_info, STFTAnalyzer


class TestAudioProcessor:
    """Test class for audio processing functions using TDD approach."""

    @pytest.fixture
    def sample_audio_path(self):
        """Fixture providing path to sample audio file."""
        return os.path.join(os.path.dirname(__file__), '..', 'samples', 'techno_bass.wav')

    def test_load_audio_valid_file(self, sample_audio_path):
        """Test loading a valid audio file."""
        audio, sr = load_audio(sample_audio_path)

        # Basic checks
        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int)
        assert sr > 0
        assert len(audio) > 0
        assert audio.ndim == 1  # Should be mono

    def test_load_audio_invalid_file(self):
        """Test loading an invalid audio file raises ValueError."""
        with pytest.raises(ValueError):
            load_audio("nonexistent_file.wav")

    def test_get_audio_info(self):
        """Test getting audio information from signal."""
        # Create synthetic audio signal
        sr = 22050
        duration = 1.0  # 1 second
        samples = int(sr * duration)
        audio = np.random.randn(samples)

        info = get_audio_info(audio, sr)

        assert info['duration'] == duration
        assert info['samples'] == samples
        assert info['sample_rate'] == sr
        assert info['channels'] == 1

    def test_get_audio_info_real_file(self, sample_audio_path):
        """Test getting audio info from real audio file."""
        audio, sr = load_audio(sample_audio_path)
        info = get_audio_info(audio, sr)

        assert info['duration'] > 0
        assert info['samples'] > 0
        assert info['sample_rate'] == sr
        assert info['channels'] == 1


class TestSTFTAnalyzer:
    """Test class for STFTAnalyzer."""

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

    def test_stft_analyzer_init(self):
        """Test STFTAnalyzer initialization."""
        analyzer = STFTAnalyzer(n_fft=2048, hop_length=512)

        assert analyzer.n_fft == 2048
        assert analyzer.hop_length == 512
        assert analyzer.window == 'hann'
        assert analyzer.stft_matrix is None

    def test_compute_stft_synthetic(self, synthetic_audio):
        """Test STFT computation on synthetic audio."""
        audio, sr = synthetic_audio
        analyzer = STFTAnalyzer(n_fft=2048, hop_length=512)

        stft_matrix = analyzer.compute_stft(audio, sr)

        # Check STFT matrix properties
        assert stft_matrix is not None
        assert stft_matrix.dtype == np.complex128
        assert stft_matrix.shape[0] == 2048 // 2 + 1  # Frequency bins
        assert stft_matrix.shape[1] > 0  # Time frames

    def test_compute_stft_real_file(self, sample_audio_path):
        """Test STFT computation on real audio file."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()

        stft_matrix = analyzer.compute_stft(audio, sr)

        assert stft_matrix is not None
        assert analyzer.magnitude_spec is not None
        assert analyzer.phase_spec is not None
        assert analyzer.magnitude_spec.shape == analyzer.phase_spec.shape

    def test_apply_noise_gate(self, sample_audio_path):
        """Test applying noise gate to STFT matrix."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)
        
        original_energy = np.sum(analyzer.magnitude_spec)
        
        analyzer.apply_noise_gate(-30)
        gated_energy = np.sum(analyzer.magnitude_spec)
        
        assert gated_energy < original_energy
        # Components below threshold should be strictly zero
        assert np.any(analyzer.magnitude_spec == 0)

    def test_get_magnitude_spectrogram(self, synthetic_audio):
        """Test magnitude spectrogram extraction."""
        audio, sr = synthetic_audio
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)

        spec = analyzer.get_magnitude_spectrogram()

        assert spec is not None
        assert spec.shape == analyzer.magnitude_spec.shape
        # Should be in dB scale (negative values expected)
        assert np.any(spec <= 0)

    def test_extract_peaks_basic(self, synthetic_audio):
        """Test basic peak extraction."""
        audio, sr = synthetic_audio
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)

        peaks = analyzer.extract_peaks(num_sinusoids=10)

        assert isinstance(peaks, list)
        assert len(peaks) > 0
        # Each peak should have required keys
        for peak in peaks:
            assert 'frequency' in peak
            assert 'amplitude' in peak
            assert 'phase' in peak
            assert 'frame_idx' in peak

    def test_extract_peaks_real_file(self, sample_audio_path):
        """Test peak extraction on real audio file."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)

        peaks = analyzer.extract_peaks(num_sinusoids=20)

        assert isinstance(peaks, list)
        assert len(peaks) > 0
        
        # Check that frequencies are in valid range
        for peak in peaks:
            assert 0 <= peak['frequency'] <= sr / 2
            assert peak['amplitude'] >= 0