import numpy as np
from typing import List, Dict, Tuple, Optional
import soundfile as sf


class SinusoidalSynthesizer:
    """
    Synthesizes audio from sinusoidal components.
    
    Reconstructs audio signals from extracted sinusoidal parameters,
    allowing control over the number of components for progressive synthesis.
    """

    def __init__(self, sr: int = 22050):
        """
        Initialize synthesizer.

        Args:
            sr: Sample rate
        """
        self.sr = sr

    def synthesize_sinusoid(self, frequency: float, amplitude: float,
                           phase: float, duration: float) -> np.ndarray:
        """
        Generate a single sinusoid.

        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude (0-1 typically)
            phase: Phase in radians
            duration: Duration in seconds

        Returns:
            Audio signal array
        """
        num_samples = int(duration * self.sr)
        t = np.arange(num_samples) / self.sr
        sinusoid = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return sinusoid

    def reconstruct_from_stft_peaks(self, sinusoids_list: List[Dict],
                                   num_sinusoids: int,
                                   original_duration: float) -> np.ndarray:
        """
        Reconstruct audio from STFT peak sinusoids.

        Args:
            sinusoids_list: List of sinusoid dicts from STFTAnalyzer.extract_peaks()
            num_sinusoids: Number of top sinusoids to use for reconstruction
            original_duration: Duration of original audio in seconds

        Returns:
            Reconstructed audio signal
        """
        if len(sinusoids_list) == 0:
            return np.zeros(int(original_duration * self.sr))

        # Sort by amplitude and take top N
        sorted_sinusoids = sorted(sinusoids_list,
                                 key=lambda x: x['amplitude'],
                                 reverse=True)
        top_sinusoids = sorted_sinusoids[:min(num_sinusoids, len(sorted_sinusoids))]

        # Initialize reconstruction
        num_samples = int(original_duration * self.sr)
        t = np.arange(num_samples) / self.sr
        reconstructed = np.zeros(num_samples)

        # Add each sinusoid
        for sinusoid in top_sinusoids:
            frequency = sinusoid['frequency']
            amplitude = sinusoid['amplitude']
            phase = sinusoid['phase']

            # Generate sinusoid for this component
            component = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            reconstructed += component

        return reconstructed

    def reconstruct_from_istft(self, stft_matrix: np.ndarray,
                              hop_length: int = 512,
                              num_sinusoids: int = None,
                              noise_gate_db: float = None,
                              keep_noise: bool = False,
                              window: str = 'hann',
                              length: int | None = None,
                              center: bool = True) -> np.ndarray:
        """
        Reconstruct audio from STFT matrix, optionally keeping only top sinusoids per frame.

        Args:
            stft_matrix: Complex STFT matrix
            hop_length: Hop length used in STFT
            num_sinusoids: If provided, zero out all but top N components per frame
            noise_gate_db: If provided, zero out components below this relative dB threshold
            keep_noise: If True, invert noise gate (keep only noise below threshold)
            window: Window type used for inverse STFT
            length: Length of output signal in samples
            center: Whether frames were centered in STFT

        Returns:
            Reconstructed audio signal
        """
        stft_copy = stft_matrix.copy()

        import librosa
        
        # Apply Spectral Noise Gate
        if noise_gate_db is not None:
            mag = np.abs(stft_copy)
            if np.max(mag) > 0:  # Prevent divide by zero warning
                mag_db = librosa.amplitude_to_db(mag, ref=np.max)
                if keep_noise:
                    mask = mag_db >= noise_gate_db
                else:
                    mask = mag_db < noise_gate_db
                stft_copy[mask] = 0

        if num_sinusoids is not None:
            # Vectorized approach: keep only top N components per frame
            if num_sinusoids < stft_copy.shape[0]:
                mags = np.abs(stft_copy)
                # Znajdź próg odcięcia (threshold) dla każdej kolumny/ramki czasowej
                thresholds = np.partition(mags, -num_sinusoids, axis=0)[-num_sinusoids, :]
                mask = mags < thresholds[np.newaxis, :]
                stft_copy[mask] = 0

        # Inverse STFT
        import librosa
        audio = librosa.istft(stft_copy, hop_length=hop_length, window=window, length=length, center=center)
        return audio

    def reconstruct_progressive(self, sinusoids_list: List[Dict],
                               original_duration: float,
                               max_sinusoids: int) -> Dict[int, np.ndarray]:
        """
        Create a dictionary of progressively reconstructed audio.

        Useful for creating animations showing how sound builds from sinusoids.

        Args:
            sinusoids_list: List of sinusoid parameters
            original_duration: Duration in seconds
            max_sinusoids: Maximum number of sinusoids to reconstruct

        Returns:
            Dict mapping number of sinusoids to reconstructed audio
        """
        result = {}

        # Sort by amplitude
        sorted_sinusoids = sorted(sinusoids_list,
                                 key=lambda x: x['amplitude'],
                                 reverse=True)

        # Start from 0
        result[0] = np.zeros(int(original_duration * self.sr))

        # Build up progressively
        num_samples = int(original_duration * self.sr)
        t = np.arange(num_samples) / self.sr
        running_sum = np.zeros(num_samples)

        for i, sinusoid in enumerate(sorted_sinusoids[:max_sinusoids]):
            frequency = sinusoid['frequency']
            amplitude = sinusoid['amplitude']
            phase = sinusoid['phase']

            component = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            running_sum += component
            result[i + 1] = running_sum.copy()

        return result

    def get_reconstruction_error(self, original: np.ndarray,
                                reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Calculate reconstruction error metrics.

        Args:
            original: Original audio signal
            reconstructed: Reconstructed audio signal

        Returns:
            Dict with error metrics (MSE, RMSE, SNR)
        """
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        original_cut = original[:min_len]
        reconstructed_cut = reconstructed[:min_len]

        # Mean Squared Error
        mse = np.mean((original_cut - reconstructed_cut) ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Signal-to-Noise Ratio (dB)
        signal_power = np.mean(original_cut ** 2)
        noise_power = np.mean((original_cut - reconstructed_cut) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return {
            'mse': mse,
            'rmse': rmse,
            'snr': snr
        }

    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio to file.

        Args:
            audio: Audio signal
            file_path: Path to save file
        """
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        sf.write(file_path, audio, self.sr)
