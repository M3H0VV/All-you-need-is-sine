import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import librosa
import scipy.ndimage
from typing import Optional, Tuple, List, Dict


class Visualizer:
    """
    Visualizes audio signals, spectrograms, and sinusoidal components.
    
    Provides interactive visualizations using Plotly for exploration
    of STFT decomposition and sinusoidal reconstruction.
    """

    def __init__(self, sr: int = 22050):
        """
        Initialize visualizer.

        Args:
            sr: Sample rate
        """
        self.sr = sr

    def plot_waveform(self, audio: np.ndarray, title: str = "Przebieg fali") -> go.Figure:
        """
        Create interactive waveform plot.

        Args:
            audio: Audio signal
            title: Plot title

        Returns:
            Plotly Figure object
        """
        time = np.arange(len(audio)) / self.sr
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=audio,
            mode='lines',
            name='Przebieg fali',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Czas (s)",
            yaxis_title="Amplituda",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig

    def plot_magnitude_spectrogram(self, magnitude_spec: np.ndarray,
                                  sr: int, hop_length: int,
                                  title: str = "Spektrogram amplitudowy (dB)",
                                  use_3d: bool = False) -> go.Figure:
        """
        Create interactive magnitude spectrogram plot.

        Args:
            magnitude_spec: Magnitude spectrogram (n_fft/2+1 x num_frames) in dB
            sr: Sample rate
            hop_length: Hop length used in STFT
            title: Plot title
            use_3d: Whether to plot as 3D surface

        Returns:
            Plotly Figure object
        """
        # Create time and frequency axes
        frames = magnitude_spec.shape[1]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=(magnitude_spec.shape[0] - 1) * 2)
        time = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=hop_length)
        
        if use_3d:
            # Wygładzenie macierzy za pomocą filtru Gaussa wyłącznie w celu uzyskania "miękkiego" widoku 3D
            smoothed_spec = scipy.ndimage.gaussian_filter(magnitude_spec, sigma=1.5)
            
            fig = go.Figure(data=[go.Surface(
                z=smoothed_spec,
                x=time,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="dB"),
                cmin=-80,
                cmax=0
            )])
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Czas (s)",
                    yaxis_title="Częstotliwość (Hz)",
                    zaxis_title="Amplituda (dB)",
                    yaxis=dict(range=[0, sr / 2]),
                    camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
                ),
                template='plotly_white',
                height=800,
                margin=dict(l=0, r=0, t=20, b=0)
            )
        else:
            fig = go.Figure(data=go.Heatmap(
                z=magnitude_spec,
                x=time,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="dB"),
                zmin=-80,
                zmax=0
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Czas (s)",
                yaxis_title="Częstotliwość (Hz)",
                yaxis_range=[0, sr / 2],
                template='plotly_white',
                height=700,
                margin=dict(l=60, r=40, t=60, b=60)
            )
        
        return fig

    def plot_phase_spectrogram(self, phase_spec: np.ndarray,
                              sr: int, hop_length: int,
                              title: str = "Spektrogram fazowy",
                              use_3d: bool = False) -> go.Figure:
        """
        Create interactive phase spectrogram plot.

        Args:
            phase_spec: Phase spectrogram (n_fft/2+1 x num_frames)
            sr: Sample rate
            hop_length: Hop length used in STFT
            title: Plot title
            use_3d: Whether to plot as 3D surface

        Returns:
            Plotly Figure object
        """
        # Create time and frequency axes
        frames = phase_spec.shape[1]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=(phase_spec.shape[0] - 1) * 2)
        time = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=hop_length)
        
        colorscale = [
            [0.00, '#000000'],  # -max
            [0.25, '#555555'],
            [0.40, '#AAAAAA'],
            [0.50, '#FFFFFF'],  # 0
            [0.60, '#AAAAAA'],
            [0.75, '#555555'],
            [1.00, '#000000']   # +max
        ]
        
        if use_3d:
            fig = go.Figure(data=[go.Surface(
                z=phase_spec,
                x=time,
                y=freqs,
                colorscale=colorscale,
                colorbar=dict(title="Radiany")
            )])
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Czas (s)",
                    yaxis_title="Częstotliwość (Hz)",
                    zaxis_title="Faza (rad)",
                    yaxis=dict(range=[0, sr / 2]),
                    camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
                ),
                template='plotly_white',
                height=800,
                margin=dict(l=0, r=0, t=20, b=0)
            )
        else:
            fig = go.Figure(data=go.Heatmap(
                z=phase_spec,
                x=time,
                y=freqs,
                colorscale=colorscale,
                colorbar=dict(title="Radiany")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Czas (s)",
                yaxis_title="Częstotliwość (Hz)",
                yaxis_range=[0, sr / 2],
                template='plotly_white',
                height=700,
                margin=dict(l=60, r=40, t=60, b=60)
            )
        
        return fig

    def plot_comparison(self, original: np.ndarray,
                       reconstructed: np.ndarray,
                       title: str = "Oryginał vs Rekonstrukcja") -> go.Figure:
        """
        Create side-by-side comparison of original and reconstructed waveforms.

        Args:
            original: Original audio signal
            reconstructed: Reconstructed audio signal
            title: Plot title

        Returns:
            Plotly Figure object
        """
        time = np.arange(len(original)) / self.sr
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=("Oryginał", "Rekonstrukcja"),
            shared_xaxes=True,
            vertical_spacing=0.12
        )
        
        # Original signal
        fig.add_trace(
            go.Scatter(x=time, y=original, mode='lines',
                      name='Oryginał',
                      line=dict(color='#1f77b4', width=1)),
            row=1, col=1
        )
        
        # Reconstructed signal
        fig.add_trace(
            go.Scatter(x=time, y=reconstructed, mode='lines',
                      name='Rekonstrukcja',
                      line=dict(color='#ff7f0e', width=1)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Czas (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplituda", row=1, col=1)
        fig.update_yaxes(title_text="Amplituda", row=2, col=1)
        
        fig.update_layout(
            title=title,
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig

    def plot_triple_comparison(self, original: np.ndarray,
                               denoised: np.ndarray,
                               noise: np.ndarray,
                               title: str = "Oryginał vs Odszumiony vs Szum") -> go.Figure:
        """
        Create 3-panel comparison of original, denoised and noise waveforms.
        """
        time = np.arange(len(original)) / self.sr
        
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=("Oryginał", "Odszumiony dźwięk", "Wycięty szum"),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=original, mode='lines', name='Oryginał',
                      line=dict(color='#1f77b4', width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=denoised, mode='lines', name='Odszumiony',
                      line=dict(color='#ff7f0e', width=1)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=noise, mode='lines', name='Szum',
                      line=dict(color='#2ca02c', width=1)),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Czas (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplituda", row=1, col=1)
        fig.update_yaxes(title_text="Amplituda", row=2, col=1)
        fig.update_yaxes(title_text="Amplituda", row=3, col=1)
        
        fig.update_layout(
            title=title,
            hovermode='x unified',
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        return fig

    def plot_frequency_spectrum(self, magnitude_db_frame: np.ndarray,
                               sr: int, n_fft: int,
                               title: str = "Widmo częstotliwości (dB)") -> go.Figure:
        """
        Create interactive 1D frequency spectrum plot from a single dB frame.

        Args:
            magnitude_db_frame: 1D array of magnitude values in dB.
            sr: Sample rate.
            n_fft: FFT size used to generate the frame.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs,
            y=magnitude_db_frame,
            mode='lines',
            name='Widmo',
            line=dict(color='#1f77b4', width=1.5)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Częstotliwość (Hz)",
            yaxis_title="Amplituda (dB)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            xaxis_range=[0, sr / 2]
        )

        return fig

    def plot_sinusoids_overlay(self, audio: np.ndarray,
                              sinusoids_list: List[Dict],
                              num_sinusoids: int,
                              title: str = "Nakładka najważniejszych sinusoid") -> go.Figure:
        """
        Plot original signal with top sinusoidal components overlaid.

        Args:
            audio: Original audio signal
            sinusoids_list: List of sinusoid parameters from STFTAnalyzer.extract_peaks()
            num_sinusoids: Number of top sinusoids to display
            title: Plot title

        Returns:
            Plotly Figure object
        """
        time = np.arange(len(audio)) / self.sr
        
        fig = go.Figure()
        
        # Original signal (thin, light)
        fig.add_trace(go.Scatter(
            x=time, y=audio, mode='lines',
            name='Oryginał',
            line=dict(color='lightgray', width=1),
            opacity=0.7
        ))
        
        # Add reconstructed from top sinusoids
        if len(sinusoids_list) > 0:
            # Sort by amplitude and take top N
            sorted_sinusoids = sorted(sinusoids_list, 
                                     key=lambda x: x['amplitude'], 
                                     reverse=True)
            top_sinusoids = sorted_sinusoids[:min(num_sinusoids, len(sorted_sinusoids))]
            
            # Reconstruct from top sinusoids
            reconstructed = np.zeros_like(audio)
            for sinusoid in top_sinusoids:
                freq = sinusoid['frequency']
                amp = sinusoid['amplitude']
                phase = sinusoid['phase']
                reconstructed += amp * np.sin(2 * np.pi * freq * time + phase)
            
            fig.add_trace(go.Scatter(
                x=time, y=reconstructed, mode='lines',
                name=f'Rekonstrukcja (top {num_sinusoids})',
                line=dict(color='#ff7f0e', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Czas (s)",
            yaxis_title="Amplituda",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig

    def plot_frequency_peaks(self, sinusoids_list: List[Dict],
                           top_n: int = 20,
                           title: str = "Najważniejsze piki częstotliwości") -> go.Figure:
        """
        Create bar plot of top frequency peaks.

        Args:
            sinusoids_list: List of sinusoid parameters
            top_n: Number of top peaks to display
            title: Plot title

        Returns:
            Plotly Figure object
        """
        if len(sinusoids_list) == 0:
            fig = go.Figure()
            fig.add_annotation(text="Brak wyodrębnionych sinusoid")
            return fig
        
        # Group by frequency to avoid duplicate peaks from different time frames
        unique_peaks = {}
        for s in sinusoids_list:
            freq_key = round(s['frequency'], 2)
            if freq_key not in unique_peaks or s['amplitude'] > unique_peaks[freq_key]['amplitude']:
                unique_peaks[freq_key] = s
                
        # Sort unique peaks by amplitude
        sorted_sinusoids = sorted(unique_peaks.values(),
                                 key=lambda x: x['amplitude'],
                                 reverse=True)
        top_sinusoids = sorted_sinusoids[:min(top_n, len(sorted_sinusoids))]
        
        frequencies = [s['frequency'] for s in top_sinusoids]
        amplitudes = [s['amplitude'] for s in top_sinusoids]
        
        fig = go.Figure(data=[
            go.Bar(x=frequencies, y=amplitudes,
                  marker_color='#1f77b4')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Częstotliwość (Hz)",
            yaxis_title="Amplituda",
            template='plotly_white',
            height=400
        )
        
        return fig
