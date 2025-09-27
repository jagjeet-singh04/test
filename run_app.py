"""
Script to run the Streamlit application
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting Customer Churn & Sales Forecasting Dashboard...")
        print("📊 Open your browser and go to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the application")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("🔧 Setting up Customer Churn & Sales Forecasting Dashboard...")
    
    # Install requirements
    if install_requirements():
        # Run the app
        run_streamlit_app()
    else:
        print("❌ Failed to install requirements. Please check your Python environment.")
