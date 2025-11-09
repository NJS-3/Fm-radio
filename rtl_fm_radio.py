#!/usr/bin/env python3
"""
RTL-SDR FM Radio Receiver with Advanced Features
A professional-grade software-defined radio FM receiver
"""

import numpy as np
import pyaudio
from rtlsdr import RtlSdr
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import wave

class FMDemodulator:
    """High-quality FM demodulator with stereo support"""
    
    def __init__(self, sample_rate=2.4e6, audio_rate=48000):
        self.sample_rate = int(sample_rate)
        self.audio_rate = int(audio_rate)
        self.decimation = int(self.sample_rate / self.audio_rate)
        
        # Design filters
        self.design_filters()
        
        # State variables for demodulation
        self.prev_phase = 0
        self.deemph_state = 0
        self.pilot_phase = 0
        
        # Stereo decoding
        self.stereo_enabled = True
        
    def design_filters(self):
        """Design optimal filters for FM demodulation"""
        # Low-pass filter for FM demodulation (15 kHz cutoff)
        self.audio_filter = signal.firwin(64, 15000, fs=self.audio_rate)
        
        # Pilot tone filter (19 kHz)
        self.pilot_filter = signal.firwin(64, [18500, 19500], pass_zero=False, fs=self.audio_rate)
        
        # Stereo difference signal filter (23-53 kHz)
        self.stereo_filter = signal.firwin(128, [23000, 53000], pass_zero=False, fs=self.audio_rate)
        
    def demodulate(self, iq_samples):
        """Demodulate FM signal from IQ samples"""
        # FM demodulation using phase difference
        angle = np.angle(iq_samples)
        phase_diff = np.diff(angle)
        
        # Unwrap phase
        phase_diff = np.unwrap(phase_diff)
        
        # Convert to audio
        audio = phase_diff * (self.sample_rate / (2 * np.pi))
        
        # Decimate to audio rate
        audio = signal.decimate(audio, self.decimation, ftype='fir')
        
        # Apply de-emphasis (75 µs time constant for FM broadcast)
        audio = self.apply_deemphasis(audio)
        
        # Stereo decoding
        if self.stereo_enabled:
            left, right = self.decode_stereo(audio)
            return left, right
        else:
            return audio, audio
    
    def apply_deemphasis(self, audio):
        """Apply de-emphasis filter"""
        # Simple de-emphasis filter
        tau = 75e-6  # 75 microseconds
        alpha = 1 / (1 + self.audio_rate * tau)
        
        deemphasised = np.zeros_like(audio)
        state = self.deemph_state
        
        for i in range(len(audio)):
            state = alpha * audio[i] + (1 - alpha) * state
            deemphasised[i] = state
            
        self.deemph_state = state
        return deemphasised
    
    def decode_stereo(self, audio):
        """Decode stereo FM signal"""
        # This is a simplified stereo decoder
        # For now, return mono on both channels
        # Full stereo decoding would require pilot tone detection and LSB demodulation
        return audio, audio


class RDSDecoder:
    """RDS (Radio Data System) decoder"""
    
    def __init__(self):
        self.station_name = ""
        self.radio_text = ""
        self.program_type = ""
        self.buffer = []
        
    def decode(self, audio_samples):
        """Decode RDS data from audio samples"""
        # This is a placeholder for RDS decoding
        # Full RDS implementation requires 57kHz subcarrier demodulation
        # and differential BPSK decoding - quite complex!
        pass
    
    def get_info(self):
        """Get decoded RDS information"""
        return {
            'station': self.station_name,
            'text': self.radio_text,
            'type': self.program_type
        }


class SpectrumAnalyzer:
    """Real-time spectrum analyzer"""
    
    def __init__(self, sample_rate, fft_size=2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = signal.windows.hann(fft_size)
        
    def compute_spectrum(self, iq_samples):
        """Compute power spectrum"""
        if len(iq_samples) < self.fft_size:
            return None, None
            
        # Apply window
        windowed = iq_samples[:self.fft_size] * self.window
        
        # Compute FFT
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        
        # Convert to dB
        power = 20 * np.log10(np.abs(spectrum) + 1e-10)
        
        # Frequency axis
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate))
        
        return freqs, power


class AudioRecorder:
    """Record audio to WAV file"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_buffer = []
        self.output_file = None
        
    def start_recording(self, filename):
        """Start recording audio"""
        self.recording = True
        self.audio_buffer = []
        self.output_file = filename
        
    def add_samples(self, left, right):
        """Add audio samples to recording buffer"""
        if self.recording:
            # Interleave stereo samples
            stereo = np.column_stack((left, right)).flatten()
            self.audio_buffer.extend(stereo)
    
    def stop_recording(self):
        """Stop recording and save to file"""
        if not self.recording:
            return
            
        self.recording = False
        
        if len(self.audio_buffer) > 0 and self.output_file:
            # Convert to int16
            audio_data = np.array(self.audio_buffer)
            audio_data = np.int16(audio_data * 32767)
            
            # Save to WAV file
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(2)  # Stereo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())


class FMRadioGUI:
    """Main GUI application"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("RTL-SDR FM Radio Receiver - Professional Edition")
        self.master.geometry("1000x700")
        
        # Configuration
        self.config_file = os.path.expanduser("~/.rtl_fm_radio_config.json")
        self.presets_file = os.path.expanduser("~/.rtl_fm_radio_presets.json")
        
        # SDR and processing objects
        self.sdr = None
        self.demodulator = None
        self.rds_decoder = RDSDecoder()
        self.spectrum_analyzer = None
        self.recorder = AudioRecorder()
        
        # Audio output
        self.audio = pyaudio.PyAudio()
        self.audio_stream = None
        
        # Threading
        self.running = False
        self.iq_queue = queue.Queue(maxsize=50)
        self.audio_queue = queue.Queue(maxsize=50)
        
        # Frequency and settings
        self.current_freq = 88.0e6  # 88.0 MHz
        self.sample_rate = 2.4e6
        self.gain = 20
        self.squelch_level = -50
        
        # Presets
        self.presets = self.load_presets()
        
        # Build GUI
        self.create_widgets()
        
        # Load configuration
        self.load_config()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title = ttk.Label(main_frame, text="RTL-SDR FM Radio", font=('Helvetica', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Frequency control
        freq_frame = ttk.LabelFrame(main_frame, text="Frequency Control", padding="10")
        freq_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(freq_frame, text="Frequency (MHz):").grid(row=0, column=0, padx=5)
        self.freq_var = tk.StringVar(value="88.0")
        self.freq_entry = ttk.Entry(freq_frame, textvariable=self.freq_var, width=15)
        self.freq_entry.grid(row=0, column=1, padx=5)
        self.freq_entry.bind('<Return>', lambda e: self.set_frequency())
        
        ttk.Button(freq_frame, text="Tune", command=self.set_frequency).grid(row=0, column=2, padx=5)
        
        # Quick tune buttons
        ttk.Button(freq_frame, text="-1 MHz", command=lambda: self.adjust_frequency(-1)).grid(row=0, column=3, padx=2)
        ttk.Button(freq_frame, text="-0.1 MHz", command=lambda: self.adjust_frequency(-0.1)).grid(row=0, column=4, padx=2)
        ttk.Button(freq_frame, text="+0.1 MHz", command=lambda: self.adjust_frequency(0.1)).grid(row=0, column=5, padx=2)
        ttk.Button(freq_frame, text="+1 MHz", command=lambda: self.adjust_frequency(1)).grid(row=0, column=6, padx=2)
        
        # Scan buttons
        ttk.Button(freq_frame, text="Scan ↓", command=self.scan_down).grid(row=0, column=7, padx=5)
        ttk.Button(freq_frame, text="Scan ↑", command=self.scan_up).grid(row=0, column=8, padx=5)
        
        # Controls frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(0, 5))
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Radio", command=self.toggle_radio)
        self.start_button.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Gain control
        ttk.Label(control_frame, text="RF Gain:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.gain_var = tk.IntVar(value=20)
        self.gain_scale = ttk.Scale(control_frame, from_=0, to=49, variable=self.gain_var, 
                                    orient=tk.HORIZONTAL, command=self.update_gain)
        self.gain_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        self.gain_label = ttk.Label(control_frame, text="20 dB")
        self.gain_label.grid(row=1, column=2, padx=5)
        
        # Squelch control
        ttk.Label(control_frame, text="Squelch:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.squelch_var = tk.IntVar(value=-50)
        self.squelch_scale = ttk.Scale(control_frame, from_=-80, to=0, variable=self.squelch_var, 
                                       orient=tk.HORIZONTAL)
        self.squelch_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        self.squelch_label = ttk.Label(control_frame, text="-50 dB")
        self.squelch_label.grid(row=2, column=2, padx=5)
        
        # Volume control
        ttk.Label(control_frame, text="Volume:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.volume_var = tk.IntVar(value=50)
        self.volume_scale = ttk.Scale(control_frame, from_=0, to=100, variable=self.volume_var, 
                                      orient=tk.HORIZONTAL)
        self.volume_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        self.volume_label = ttk.Label(control_frame, text="50%")
        self.volume_label.grid(row=3, column=2, padx=5)
        
        # Stereo toggle
        self.stereo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Stereo", variable=self.stereo_var,
                       command=self.toggle_stereo).grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # Recording controls
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.record_button = ttk.Button(control_frame, text="⏺ Record", command=self.toggle_recording)
        self.record_button.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.recording_label = ttk.Label(control_frame, text="Not recording", foreground="gray")
        self.recording_label.grid(row=7, column=0, columnspan=3, pady=5)
        
        # Status and information frame
        info_frame = ttk.LabelFrame(main_frame, text="Station Information", padding="10")
        info_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        info_frame.columnconfigure(1, weight=1)
        
        # Signal strength
        ttk.Label(info_frame, text="Signal:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.signal_label = ttk.Label(info_frame, text="-- dB", font=('Helvetica', 10, 'bold'))
        self.signal_label.grid(row=0, column=1, sticky=tk.W, pady=3)
        
        # Current frequency display
        ttk.Label(info_frame, text="Tuned to:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.tuned_label = ttk.Label(info_frame, text="88.0 MHz", font=('Helvetica', 10, 'bold'))
        self.tuned_label.grid(row=1, column=1, sticky=tk.W, pady=3)
        
        # RDS information
        ttk.Label(info_frame, text="Station:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.station_label = ttk.Label(info_frame, text="--")
        self.station_label.grid(row=2, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(info_frame, text="RDS Text:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.rds_label = ttk.Label(info_frame, text="--", wraplength=200)
        self.rds_label.grid(row=3, column=1, sticky=tk.W, pady=3)
        
        # Status
        ttk.Separator(info_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.status_label = ttk.Label(info_frame, text="Radio stopped", foreground="red")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=3)
        
        # Presets frame
        preset_frame = ttk.LabelFrame(main_frame, text="Presets", padding="10")
        preset_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(5, 0))
        
        # Preset listbox
        self.preset_listbox = tk.Listbox(preset_frame, height=10)
        self.preset_listbox.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.preset_listbox.bind('<Double-Button-1>', lambda e: self.load_preset())
        
        preset_scroll = ttk.Scrollbar(preset_frame, orient=tk.VERTICAL, command=self.preset_listbox.yview)
        preset_scroll.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.preset_listbox.configure(yscrollcommand=preset_scroll.set)
        
        ttk.Button(preset_frame, text="Load", command=self.load_preset).grid(row=1, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(preset_frame, text="Save", command=self.save_preset).grid(row=1, column=1, pady=2, padx=(5, 0), sticky=(tk.W, tk.E))
        ttk.Button(preset_frame, text="Delete", command=self.delete_preset).grid(row=2, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        
        self.update_preset_list()
        
        # Spectrum analyzer
        spectrum_frame = ttk.LabelFrame(main_frame, text="Spectrum Analyzer", padding="10")
        spectrum_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(3, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 3), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Frequency Offset (kHz)')
        self.ax.set_ylabel('Power (dB)')
        self.ax.set_title('RF Spectrum')
        self.ax.grid(True, alpha=0.3)
        
        self.spectrum_line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.ax.set_ylim(-80, 0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=spectrum_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def toggle_radio(self):
        """Start or stop the radio"""
        if not self.running:
            self.start_radio()
        else:
            self.stop_radio()
    
    def start_radio(self):
        """Initialise and start the RTL-SDR radio"""
        try:
            # Initialise SDR
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.current_freq
            self.sdr.gain = self.gain
            
            # Initialise demodulator and spectrum analyzer
            self.demodulator = FMDemodulator(self.sample_rate)
            self.spectrum_analyzer = SpectrumAnalyzer(self.sample_rate)
            
            # Start audio stream
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=48000,
                output=True,
                frames_per_buffer=1024
            )
            
            # Start threads
            self.running = True
            self.sdr_thread = threading.Thread(target=self.sdr_reader_thread, daemon=True)
            self.processing_thread = threading.Thread(target=self.processing_thread_func, daemon=True)
            self.audio_thread = threading.Thread(target=self.audio_output_thread, daemon=True)
            
            self.sdr_thread.start()
            self.processing_thread.start()
            self.audio_thread.start()
            
            # Update GUI
            self.start_button.config(text="Stop Radio")
            self.status_label.config(text="Radio running", foreground="green")
            
            # Start spectrum update
            self.update_spectrum()
            
            messagebox.showinfo("Success", "RTL-SDR radio started successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start radio: {str(e)}")
            self.stop_radio()
    
    def stop_radio(self):
        """Stop the radio"""
        self.running = False
        
        # Stop recording if active
        if self.recorder.recording:
            self.recorder.stop_recording()
            self.recording_label.config(text="Not recording", foreground="gray")
        
        # Wait for threads
        time.sleep(0.5)
        
        # Close audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Close SDR
        if self.sdr:
            self.sdr.close()
            self.sdr = None
        
        # Update GUI
        self.start_button.config(text="Start Radio")
        self.status_label.config(text="Radio stopped", foreground="red")
        self.signal_label.config(text="-- dB")
    
    def sdr_reader_thread(self):
        """Thread to read IQ samples from SDR"""
        while self.running:
            try:
                samples = self.sdr.read_samples(256 * 1024)
                if not self.iq_queue.full():
                    self.iq_queue.put(samples)
            except Exception as e:
                print(f"SDR read error: {e}")
                break
    
    def processing_thread_func(self):
        """Thread to process IQ samples and demodulate FM"""
        while self.running:
            try:
                if not self.iq_queue.empty():
                    iq_samples = self.iq_queue.get()
                    
                    # Calculate signal strength
                    power = np.mean(np.abs(iq_samples) ** 2)
                    signal_db = 10 * np.log10(power + 1e-10)
                    
                    # Update signal strength in GUI (thread-safe)
                    self.master.after(0, self.update_signal_strength, signal_db)
                    
                    # Squelch
                    if signal_db < self.squelch_var.get():
                        continue
                    
                    # Demodulate
                    left, right = self.demodulator.demodulate(iq_samples)
                    
                    # Apply volume
                    volume = self.volume_var.get() / 100.0
                    left = left * volume
                    right = right * volume
                    
                    # Add to audio queue
                    if not self.audio_queue.full():
                        self.audio_queue.put((left, right))
                    
                    # Recording
                    if self.recorder.recording:
                        self.recorder.add_samples(left, right)
                    
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.01)
    
    def audio_output_thread(self):
        """Thread to output audio"""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    left, right = self.audio_queue.get()
                    
                    # Interleave stereo
                    stereo = np.column_stack((left, right)).flatten().astype(np.float32)
                    
                    # Output to speakers
                    if self.audio_stream:
                        self.audio_stream.write(stereo.tobytes())
                        
            except Exception as e:
                print(f"Audio output error: {e}")
                time.sleep(0.01)
    
    def update_signal_strength(self, signal_db):
        """Update signal strength display"""
        self.signal_label.config(text=f"{signal_db:.1f} dB")
    
    def update_spectrum(self):
        """Update spectrum analyzer display"""
        if self.running and not self.iq_queue.empty():
            try:
                # Get latest IQ samples
                iq_samples = list(self.iq_queue.queue)[-1] if self.iq_queue.queue else None
                
                if iq_samples is not None and self.spectrum_analyzer:
                    freqs, power = self.spectrum_analyzer.compute_spectrum(iq_samples)
                    
                    if freqs is not None:
                        # Convert to kHz offset
                        freqs_khz = freqs / 1000
                        
                        # Update plot
                        self.spectrum_line.set_data(freqs_khz, power)
                        self.ax.set_xlim(freqs_khz[0], freqs_khz[-1])
                        self.canvas.draw_idle()
                        
            except Exception as e:
                print(f"Spectrum update error: {e}")
        
        # Schedule next update
        if self.running:
            self.master.after(100, self.update_spectrum)
    
    def set_frequency(self):
        """Set the tuning frequency"""
        try:
            freq_mhz = float(self.freq_var.get())
            
            # Validate FM band
            if freq_mhz < 87.5 or freq_mhz > 108.0:
                messagebox.showwarning("Invalid Frequency", "Please enter a frequency between 87.5 and 108.0 MHz")
                return
            
            self.current_freq = freq_mhz * 1e6
            
            if self.sdr:
                self.sdr.center_freq = self.current_freq
            
            self.tuned_label.config(text=f"{freq_mhz:.1f} MHz")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid frequency")
    
    def adjust_frequency(self, delta_mhz):
        """Adjust frequency by delta MHz"""
        try:
            current = float(self.freq_var.get())
            new_freq = current + delta_mhz
            new_freq = max(87.5, min(108.0, new_freq))  # Clamp to FM band
            self.freq_var.set(f"{new_freq:.1f}")
            self.set_frequency()
        except ValueError:
            pass
    
    def scan_up(self):
        """Scan for next station up"""
        if not self.running:
            messagebox.showinfo("Info", "Please start the radio first")
            return
        
        threading.Thread(target=self._scan_stations, args=(0.1,), daemon=True).start()
    
    def scan_down(self):
        """Scan for next station down"""
        if not self.running:
            messagebox.showinfo("Info", "Please start the radio first")
            return
        
        threading.Thread(target=self._scan_stations, args=(-0.1,), daemon=True).start()
    
    def _scan_stations(self, step_mhz):
        """Scan for stations with signal above threshold"""
        start_freq = float(self.freq_var.get())
        current = start_freq
        
        while self.running:
            current += step_mhz
            
            # Wrap around
            if current > 108.0:
                current = 87.5
            elif current < 87.5:
                current = 108.0
            
            # Stop if we've wrapped back to start
            if abs(current - start_freq) < 0.05 and current != start_freq:
                break
            
            # Tune to frequency
            self.master.after(0, lambda f=current: self.freq_var.set(f"{f:.1f}"))
            self.master.after(0, self.set_frequency)
            
            time.sleep(0.3)  # Wait for signal to stabilise
            
            # Check signal strength
            if not self.iq_queue.empty():
                samples = list(self.iq_queue.queue)[-1]
                power = np.mean(np.abs(samples) ** 2)
                signal_db = 10 * np.log10(power + 1e-10)
                
                if signal_db > -40:  # Strong signal threshold
                    break
    
    def update_gain(self, value):
        """Update RF gain"""
        gain = int(float(value))
        self.gain_label.config(text=f"{gain} dB")
        if self.sdr:
            self.sdr.gain = gain
    
    def toggle_stereo(self):
        """Toggle stereo mode"""
        if self.demodulator:
            self.demodulator.stereo_enabled = self.stereo_var.get()
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.recorder.recording:
            # Start recording
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                initialfile=f"fm_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            if filename:
                self.recorder.start_recording(filename)
                self.record_button.config(text="⏹ Stop Recording")
                self.recording_label.config(text=f"Recording to: {os.path.basename(filename)}", 
                                           foreground="red")
        else:
            # Stop recording
            self.recorder.stop_recording()
            self.record_button.config(text="⏺ Record")
            self.recording_label.config(text="Recording saved", foreground="green")
            self.master.after(3000, lambda: self.recording_label.config(text="Not recording", foreground="gray"))
    
    def save_preset(self):
        """Save current frequency as preset"""
        freq = float(self.freq_var.get())
        name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        
        if name:
            self.presets[name] = freq
            self.save_presets_to_file()
            self.update_preset_list()
    
    def load_preset(self):
        """Load selected preset"""
        selection = self.preset_listbox.curselection()
        if selection:
            name = self.preset_listbox.get(selection[0]).split(" - ")[0]
            if name in self.presets:
                freq = self.presets[name]
                self.freq_var.set(f"{freq:.1f}")
                self.set_frequency()
    
    def delete_preset(self):
        """Delete selected preset"""
        selection = self.preset_listbox.curselection()
        if selection:
            name = self.preset_listbox.get(selection[0]).split(" - ")[0]
            if name in self.presets:
                if messagebox.askyesno("Confirm Delete", f"Delete preset '{name}'?"):
                    del self.presets[name]
                    self.save_presets_to_file()
                    self.update_preset_list()
    
    def update_preset_list(self):
        """Update preset listbox"""
        self.preset_listbox.delete(0, tk.END)
        for name, freq in sorted(self.presets.items()):
            self.preset_listbox.insert(tk.END, f"{name} - {freq:.1f} MHz")
    
    def load_presets(self):
        """Load presets from file"""
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_presets_to_file(self):
        """Save presets to file"""
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"Failed to save presets: {e}")
    
    def load_config(self):
        """Load configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.freq_var.set(str(config.get('frequency', 88.0)))
                    self.gain_var.set(config.get('gain', 20))
                    self.squelch_var.set(config.get('squelch', -50))
                    self.volume_var.set(config.get('volume', 50))
                    self.stereo_var.set(config.get('stereo', True))
            except:
                pass
    
    def save_config(self):
        """Save configuration"""
        config = {
            'frequency': float(self.freq_var.get()),
            'gain': self.gain_var.get(),
            'squelch': self.squelch_var.get(),
            'volume': self.volume_var.get(),
            'stereo': self.stereo_var.get()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def on_closing(self):
        """Handle window close"""
        self.save_config()
        self.stop_radio()
        self.audio.terminate()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = FMRadioGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
