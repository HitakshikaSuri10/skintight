#!/usr/bin/env python3
"""
SkinSight - Startup Script

This script provides an easy way to run the skin detection application
with proper configuration and error handling.

Usage:
    python run.py
    python run.py --debug
    python run.py --port 8080
"""

import argparse
import sys
import os
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['uploads', 'static', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created/verified")

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description='SkinSight - Skin Problem Detection')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run on (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("🚀 Starting SkinSight...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Import and run the app
    try:
        from app import app
        
        print(f"🌐 Starting server on http://{args.host}:{args.port}")
        print("📱 Open your browser and navigate to the URL above")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(
            debug=args.debug,
            host=args.host,
            port=args.port,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Make sure you're in the correct directory and all files are present.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 