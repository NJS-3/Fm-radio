### Core Features
- **High-Quality FM Demodulation** - Proper FM demodulation with de-emphasis filtering
- **Stereo Support** - True stereo decoding (mono/stereo switchable)
- **Real-Time Spectrum Analyzer** - See the RF spectrum in real-time with FFT visualisation
- **Auto Station Scanning** - Scan up/down to find stations automatically
- **Recording Capability** - Record audio to WAV files with timestamp
- **Signal Strength Meter** - Real-time signal level monitoring
- **Squelch Control** - Automatic noise suppression when signal is weak
- **Variable RF Gain** - Adjust receiver sensitivity (0-49 dB)
- **Volume Control** - Independent volume control
- **Preset Management** - Save and recall favourite stations
- **Professional GUI** - Clean, intuitive Tkinter interface

### Advanced Features
- **Multi-threaded Architecture** - Separate threads for SDR reading, processing, and audio output
- **Configurable Sample Rates** - Optimised for 2.4 MSPS
- **De-emphasis Filtering** - Proper 75µs time constant for FM broadcast
- **Queue-based Audio Pipeline** - Smooth, glitch-free audio playback
- **Persistent Configuration** - Saves settings between sessions
- **Fine-Tuning Controls** - ±1 MHz and ±0.1 MHz quick tune buttons

##  Requirements

### Hardware
- RTL-SDR USB dongle (RTL2832U-based)
  - DVB-T dongles like RTL-SDR Blog V3, NooElec, or similar
  - Proper FM antenna (telescopic or dipole recommended)

### Software
- Python 3.7+
- RTL-SDR drivers installed
- Audio system (ALSA/PulseAudio on Linux, CoreAudio on macOS, DirectSound on Windows)

## Advanced Features That Don't Exist on Traditional Radios

### 1. **Real-Time Spectrum Visualization**
```
Traditional Radio: [Silence] "Is there a station here?"
SDR Radio:         [Shows entire FM band visually]
                   "Here are ALL stations with signal strength!"
```

### 2. **Precision Tuning**
```
Traditional:  *twists knob* "Was that 97.5 or 97.6?"
SDR:          Types "97.6" - locked on perfectly
```

### 3. **Recording with Metadata**
```
Traditional:  *scrambles for tape recorder*
SDR:          Click "Record" - timestamped WAV file saved
```

### 4. **Intelligent Scanning**
```
Traditional:  Scans through static... static... static...
SDR:          Scans and STOPS on strong signals only
```

### 5. **Unlimited Presets with Names**
```
Traditional:  Button 1, Button 2, Button 3...
SDR:          "BBC Radio 1", "Jazz FM", "Capital London"...
```

### 6. **Professional Signal Analysis**
```
Traditional:  "Signal's a bit weak"
SDR:          "Signal: -42.3 dB, adequate for stereo reception"
```

### 7. **Software Updates**
```
Traditional:  Hardware is hardware forever
SDR:          Download new features, modes, improvements
```

### 8. **Cross-Platform**
```
Traditional:  This specific radio
SDR:          Same software on Windows, Mac, Linux, Raspberry Pi
```

### 9. **Multiple Simultaneous Receivers** (Future)
```
Traditional:  One radio = one station
SDR:          Multiple virtual receivers from one dongle
```

### 10. **Full Digital Signal Processing**
```
Traditional:  Analogue circuits, drift, noise
SDR:          Perfect digital filtering, no drift, adaptable
