import numpy as np
import librosa
import soundfile as sf
import warnings
from typing import Tuple, Optional, List, Dict
from scipy import signal


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to mono signal, preserving original sample rate.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple[audio_signal, sample_rate]
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            # Load audio with librosa, sr=None preserves the original sample rate
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        return audio, sample_rate
    except Exception as e:
        if "NoBackendError" in e.__class__.__name__:
            raise ValueError("Brak zainstalowanego pakietu FFmpeg. Aby wczytywać pliki z dyktafonu (.amr) lub inne formaty, zainstaluj FFmpeg w systemie (i dodaj do PATH) lub użyj klasycznego formatu .wav.")
            
        raise ValueError(f"Error loading audio file: {e}")


def get_audio_info(audio: np.ndarray, sr: int) -> dict:
    """
    Return basic information about audio signal.

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        Dict with audio information
    """
    duration = len(audio) / sr
    return {
        'duration': duration,
        'samples': len(audio),
        'sample_rate': sr,
        'channels': 1  # always mono
    }


class STFTAnalyzer:
    """
    Analyzes audio using STFT and extracts sinusoidal components.
    
    This class performs Short-Time Fourier Transform decomposition
    and allows extraction of top-N sinusoidal components for reconstruction.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, 
                 window: str = 'hann', center: bool = True):
        """
        Initialize STFT analyzer.

        Args:
            n_fft: FFT size (default 2048)
            hop_length: Number of samples between frames (default 512)
            window: Window function type (default 'hann')
            center: Whether to center frames (default True)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center
        
        # Will store analysis results
        self.stft_matrix = None
        self.magnitude_spec = None
        self.phase_spec = None
        self.audio = None
        self.sr = None

    def compute_stft(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute STFT of audio signal.

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Complex STFT matrix (n_fft/2+1 x num_frames)
        """
        self.audio = audio
        self.sr = sr
        
        # Compute STFT
        self.stft_matrix = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center
        )
        
        # Extract magnitude and phase
        self.magnitude_spec = np.abs(self.stft_matrix)
        self.phase_spec = np.angle(self.stft_matrix)
        
        return self.stft_matrix

    def apply_noise_gate(self, noise_gate_db: float) -> None:
        """
        Zero out STFT components below a relative dB threshold.
        
        Args:
            noise_gate_db: Threshold in relative dB (e.g. -40)
        """
        if self.stft_matrix is None:
            raise ValueError("Must call compute_stft() first")
            
        mag = np.abs(self.stft_matrix)
        if np.max(mag) > 0:
            mag_db = librosa.amplitude_to_db(mag, ref=np.max)
            mask = mag_db < noise_gate_db
            self.stft_matrix[mask] = 0
            
            self.magnitude_spec = np.abs(self.stft_matrix)
            self.phase_spec = np.angle(self.stft_matrix)

    def get_magnitude_spectrogram(self) -> np.ndarray:
        """
        Get magnitude spectrogram (in dB scale).

        Returns:
            Magnitude spectrogram in dB
        """
        if self.magnitude_spec is None:
            raise ValueError("Must call compute_stft() first")
        
        # Convert to dB scale for better visualization
        spectrogram_db = librosa.power_to_db(
            self.magnitude_spec ** 2,
            ref=np.max
        )
        return spectrogram_db

    def extract_peaks(self, num_sinusoids: int = 50,
                     prominence_threshold: float = 0.01) -> List[Dict]:
        """
        Extract top-N sinusoidal peaks from spectrogram.

        Args:
            num_sinusoids: Number of peaks to extract per frame
            prominence_threshold: Minimum relative magnitude for peak

        Returns:
            List of dicts with: 'frequency', 'amplitude', 'phase', 'frame_idx'
        """
        if self.magnitude_spec is None:
            raise ValueError("Must call compute_stft() first")
        
        sinusoids = []
        num_frames = self.magnitude_spec.shape[1]
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # For each frame, find top N peaks
        for frame_idx in range(num_frames):
            magnitude_frame = self.magnitude_spec[:, frame_idx]
            phase_frame = self.phase_spec[:, frame_idx]
            
            # Find peaks
            peaks, properties = signal.find_peaks(
                magnitude_frame,
                height=prominence_threshold * np.max(magnitude_frame),
                distance=2  # Minimum distance between peaks
            )
            
            # Sort by magnitude (height)
            if len(peaks) > 0:
                sorted_idx = np.argsort(properties['peak_heights'])[::-1]
                peaks = peaks[sorted_idx]
                
                # Take top N
                top_peaks = peaks[:min(num_sinusoids, len(peaks))]
                
                for peak_bin in top_peaks:
                    sinusoids.append({
                        'frequency': freqs[peak_bin],
                        'bin': peak_bin,
                        'amplitude': magnitude_frame[peak_bin],
                        'phase': phase_frame[peak_bin],
                        'frame_idx': frame_idx,
                        'time': frame_idx * self.hop_length / self.sr
                    })
        
        return sinusoids