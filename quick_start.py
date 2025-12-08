#!/usr/bin/env python3
"""Quick start script for community detection project."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🚀 Community Detection with Graph Neural Networks - Quick Start")
    print("=" * 70)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Run example
    print("\n🔄 Running example...")
    try:
        import example
        example.main()
        print("✅ Example completed successfully")
    except Exception as e:
        print(f"❌ Example failed: {e}")
    
    # Show available commands
    print("\n📋 Available Commands:")
    print("  python train.py                    # Train with default config")
    print("  python train.py --model_type gat   # Train GAT model")
    print("  python example.py                  # Run example script")
    print("  python legacy_demo.py               # Run original Louvain demo")
    print("  streamlit run demo/app.py          # Launch interactive demo")
    print("  python -m pytest tests/            # Run tests")
    
    print("\n🎉 Setup completed! You can now:")
    print("  1. Run 'python train.py' to train a model")
    print("  2. Run 'streamlit run demo/app.py' for interactive demo")
    print("  3. Check the README.md for detailed documentation")


if __name__ == "__main__":
    main()
