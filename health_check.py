#!/usr/bin/env python3
"""
Health check script for Streamlit Cloud deployment
Ensures the FoT Fluid Dynamics app is running correctly
"""

import requests
import sys
import time
from pathlib import Path

def check_app_health(url="http://localhost:8501", timeout=30):
    """Check if the Streamlit app is responsive"""
    try:
        print(f"🔍 Checking app health at {url}...")
        
        # Try to reach the health endpoint
        response = requests.get(f"{url}/healthz", timeout=timeout)
        if response.status_code == 200:
            print("✅ Health endpoint responding")
            return True
            
    except requests.exceptions.RequestException:
        # Try the main page if health endpoint doesn't exist
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                print("✅ Main page responding")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ App not responding: {e}")
            return False
    
    return False

def check_data_files():
    """Check if essential data files exist"""
    print("📁 Checking data files...")
    
    proof_dir = Path("data/millennium_proofs")
    if proof_dir.exists():
        print("✅ Proof storage directory exists")
        
        proof_file = proof_dir / "millennium_proofs.json"
        if proof_file.exists():
            print("✅ Proof data file exists")
        else:
            print("⚠️  Proof data file missing (will be created)")
    else:
        print("⚠️  Proof directory missing (will be created)")
    
    return True

def main():
    """Run comprehensive health check"""
    print("🏥 FoT Fluid Dynamics - Health Check")
    print("=" * 50)
    
    # Check data files
    data_ok = check_data_files()
    
    # Check app responsiveness
    app_ok = check_app_health()
    
    if data_ok and app_ok:
        print("\n🎉 All systems healthy!")
        sys.exit(0)
    else:
        print("\n❌ Health check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
