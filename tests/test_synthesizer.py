import pytest
import numpy as np
import os
from src.synthesizer import SinusoidalSynthesizer
from src.audio_processor import load_audio, STFTAnalyzer


class TestSinusoidalSynthesizer:
    """Test class for SinusoidalSynthesizer."""

    @pytest.fixture
    def sample_audio_path(self):
        """Fixture providing path to sample audio file."""
        return os.path.join(os.path.dirname(__file__), '..', 'samples', 'techno_bass.wav')

    @pytest.fixture
    def synthesizer(self):
        """Fixture providing SinusoidalSynthesizer instance."""
        return SinusoidalSynthesizer(sr=22050)

    @pytest.fixture
    def sample_sinusoids(self):
        """Fixture providing sample sinusoid parameters."""
        return [
            {'frequency': 440, 'amplitude': 0.5, 'phase': 0, 'frame_idx': 0, 'time': 0},
            {'frequency': 880, 'amplitude': 0.3, 'phase': 0.5, 'frame_idx': 0, 'time': 0},
            {'frequency': 220, 'amplitude': 0.2, 'phase': 1.0, 'frame_idx': 0, 'time': 0},
        ]

    def test_synthesizer_init(self):
        """Test SinusoidalSynthesizer initialization."""
        synth = SinusoidalSynthesizer(sr=22050)
        assert synth.sr == 22050

    def test_synthesize_sinusoid(self, synthesizer):
        """Test single sinusoid generation."""
        frequency = 440  # A4
        amplitude = 0.5
        phase = 0
        duration = 1.0

        sinusoid = synthesizer.synthesize_sinusoid(
            frequency, amplitude, phase, duration
        )

        # Check output properties
        assert len(sinusoid) == 22050  # 1 second at 22050 Hz
        assert isinstance(sinusoid, np.ndarray)
        assert np.max(np.abs(sinusoid)) <= amplitude + 0.01  # Allow small numerical error

    def test_synthesize_sinusoid_different_phases(self, synthesizer):
        """Test that different phases produce different signals."""
        freq = 440
        amp = 0.5
        dur = 0.1

        sin_phase0 = synthesizer.synthesize_sinusoid(freq, amp, 0, dur)
        sin_phase_pi = synthesizer.synthesize_sinusoid(freq, amp, np.pi, dur)

        # Signals should be different
        assert not np.allclose(sin_phase0, sin_phase_pi)
        # But similar in magnitude
        assert np.allclose(np.abs(sin_phase0), np.abs(sin_phase_pi), atol=0.01)

    def test_reconstruct_from_stft_peaks_empty(self, synthesizer):
        """Test reconstruction from empty sinusoid list."""
        duration = 1.0
        audio = synthesizer.reconstruct_from_stft_peaks([], 10, duration)

        assert len(audio) == 22050  # 1 second at 22050 Hz
        assert np.allclose(audio, 0)  # Should be all zeros

    def test_reconstruct_from_stft_peaks_basic(self, synthesizer, sample_sinusoids):
        """Test basic reconstruction from sinusoid parameters."""
        duration = 1.0
        num_sinusoids = 2

        audio = synthesizer.reconstruct_from_stft_peaks(
            sample_sinusoids, num_sinusoids, duration
        )

        assert len(audio) == 22050
        assert isinstance(audio, np.ndarray)
        # Reconstructed signal should have non-zero values
        assert np.max(np.abs(audio)) > 0

    def test_reconstruct_from_stft_peaks_varying_num(self, synthesizer, sample_sinusoids):
        """Test reconstruction with different numbers of sinusoids."""
        duration = 1.0

        audio_1 = synthesizer.reconstruct_from_stft_peaks(sample_sinusoids, 1, duration)
        audio_2 = synthesizer.reconstruct_from_stft_peaks(sample_sinusoids, 2, duration)
        audio_3 = synthesizer.reconstruct_from_stft_peaks(sample_sinusoids, 3, duration)

        # Signals should be different
        assert not np.allclose(audio_1, audio_2)
        assert not np.allclose(audio_2, audio_3)

        # More sinusoids should give larger amplitudes (roughly)
        assert np.max(np.abs(audio_3)) > np.max(np.abs(audio_1))

    def test_reconstruct_progressive(self, synthesizer, sample_sinusoids):
        """Test progressive reconstruction."""
        duration = 1.0
        max_sinusoids = 3

        result = synthesizer.reconstruct_progressive(
            sample_sinusoids, duration, max_sinusoids
        )

        # Should have entries for 0 to max_sinusoids
        assert len(result) == max_sinusoids + 1
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert 3 in result

        # Check that audio grows progressively
        assert np.allclose(result[0], 0)  # 0 sinusoids = silence
        assert np.max(np.abs(result[1])) > 0  # 1 sinusoid = signal
        assert np.max(np.abs(result[3])) >= np.max(np.abs(result[1]))  # 3 >= 1

    def test_get_reconstruction_error(self, synthesizer):
        """Test reconstruction error calculation."""
        # Create test signals
        original = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
        reconstructed = original * 0.9  # 90% accuracy

        errors = synthesizer.get_reconstruction_error(original, reconstructed)

        # Check error metrics exist and have reasonable values
        assert 'mse' in errors
        assert 'rmse' in errors
        assert 'snr' in errors

        assert errors['mse'] >= 0
        assert errors['rmse'] >= 0
        assert errors['snr'] < 30  # SNR should be less than perfect (30 dB is quite good)

    def test_get_reconstruction_error_perfect(self, synthesizer):
        """Test reconstruction error with perfect reconstruction."""
        original = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
        reconstructed = original.copy()

        errors = synthesizer.get_reconstruction_error(original, reconstructed)

        # Should be nearly perfect
        assert errors['mse'] < 1e-10
        assert errors['rmse'] < 1e-5

    def test_reconstruct_from_stft_real_file(self, synthesizer, sample_audio_path):
        """Test reconstruction from real audio STFT."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        analyzer.compute_stft(audio, sr)
        sinusoids = analyzer.extract_peaks(num_sinusoids=50)

        # Get original duration
        duration = len(audio) / sr

        # Reconstruct with different numbers of sinusoids
        recon_5 = synthesizer.reconstruct_from_stft_peaks(sinusoids, 5, duration)
        recon_20 = synthesizer.reconstruct_from_stft_peaks(sinusoids, 20, duration)

        # Check that both have correct length
        assert len(recon_5) > 0
        assert len(recon_20) > 0

        # Check that reconstructed signals have expected properties
        errors_5 = synthesizer.get_reconstruction_error(audio, recon_5)
        errors_20 = synthesizer.get_reconstruction_error(audio, recon_20)

        # Error metrics should exist
        assert 'snr' in errors_5
        assert 'snr' in errors_20
        assert errors_5['rmse'] > 0
        assert errors_20['rmse'] > 0

    def test_reconstruct_from_istft_noise_gate(self, synthesizer, sample_audio_path):
        """Test ISTFT reconstruction with noise gating."""
        audio, sr = load_audio(sample_audio_path)
        analyzer = STFTAnalyzer()
        stft = analyzer.compute_stft(audio, sr)
        
        recon_no_gate = synthesizer.reconstruct_from_istft(stft, center=True)
        recon_gate = synthesizer.reconstruct_from_istft(stft, noise_gate_db=-30, center=True)
        
        assert len(recon_no_gate) > 0
        assert len(recon_gate) > 0
        # Gated audio should have less energy overall because components were zeroed
        assert np.sum(np.abs(recon_gate)) < np.sum(np.abs(recon_no_gate))
