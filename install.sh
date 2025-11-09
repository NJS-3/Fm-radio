#!/bin/bash
# RTL-SDR FM Radio Installation Script

echo "=========================================="
echo "RTL-SDR FM Radio - Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python3_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.7+ required. You have Python $python3_version"
    exit 1
fi
echo "✓ Python $python3_version detected"
echo ""

# Detect OS
echo "Detecting operating system..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "✓ Linux detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo "✓ macOS detected"
else
    OS="unknown"
    echo "⚠ Unknown OS: $OSTYPE"
fi
echo ""

# Install system dependencies
echo "Installing system dependencies..."
if [ "$OS" == "linux" ]; then
    echo "This will require sudo privileges..."
    
    # Check if apt is available
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y rtl-sdr librtlsdr-dev portaudio19-dev python3-dev
        echo "✓ System dependencies installed"
    else
        echo "⚠ apt-get not found. Please install rtl-sdr and portaudio manually."
    fi
    
elif [ "$OS" == "mac" ]; then
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        brew install librtlsdr portaudio
        echo "✓ System dependencies installed"
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    echo "   Try: pip3 install --user numpy scipy pyrtlsdr pyaudio matplotlib"
    exit 1
fi
echo ""

# Test RTL-SDR device
echo "Testing RTL-SDR device..."
if command -v rtl_test &> /dev/null; then
    echo "Running rtl_test for 2 seconds..."
    timeout 2s rtl_test 2>&1 | head -n 10
    
    if [ $? -eq 124 ]; then
        echo "✓ RTL-SDR device detected!"
    else
        echo "⚠ RTL-SDR device not detected or not responding"
        echo "   Please connect your RTL-SDR dongle and check USB connection"
    fi
else
    echo "⚠ rtl_test command not found"
fi
echo ""

# Create launcher script
echo "Creating launcher script..."
cat > launch_radio.sh << 'EOF'
#!/bin/bash
# RTL-SDR FM Radio Launcher

echo "Starting RTL-SDR FM Radio..."
python3 rtl_fm_radio.py
EOF

chmod +x launch_radio.sh
echo "✓ Launcher script created: ./launch_radio.sh"
echo ""

# Final instructions
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To start the radio:"
echo "  ./launch_radio.sh"
echo ""
echo "Or directly:"
echo "  python3 rtl_fm_radio.py"
echo ""
echo "Quick Start:"
echo "  1. Connect your RTL-SDR dongle with antenna"
echo "  2. Run the launcher script"
echo "  3. Click 'Start Radio'"
echo "  4. Enter frequency (e.g., 88.0 MHz)"
echo "  5. Enjoy!"
echo ""
echo "Documentation: See README.md"
echo "=========================================="
