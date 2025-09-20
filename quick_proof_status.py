#!/usr/bin/env python3
"""
Quick Proof Status - Shows existing proof status without regenerating
"""

import json
from pathlib import Path
from datetime import datetime

def check_proof_status():
    """Check if we have valid proofs and show status"""
    
    print("🏆 MILLENNIUM PRIZE PROOF STATUS")
    print("=" * 40)
    
    storage_dir = Path("data/millennium_proofs")
    proof_file = storage_dir / "millennium_proofs.json"
    solution_file = storage_dir / "solution_sequences.json"
    
    if not storage_dir.exists():
        print("❌ No proof storage directory found")
        print("🔧 Run: python3 generate_millennium_proof.py")
        return False
    
    if not proof_file.exists():
        print("❌ No proof files found")
        print("🔧 Run: python3 generate_millennium_proof.py")
        return False
    
    try:
        # Load and display proof status
        with open(proof_file, 'r') as f:
            proofs = json.load(f)
        
        print(f"✅ Found {len(proofs)} proof(s)")
        print("")
        
        for problem_id, proof_data in proofs.items():
            print(f"📋 Problem: {problem_id}")
            print(f"🎯 Confidence: {proof_data.get('proof_confidence', 0):.3f}")
            print(f"🏆 Solved: {proof_data.get('proof_solved', False)}")
            print(f"📅 Generated: {proof_data.get('timestamp', 'Unknown')}")
            
            cert = proof_data.get('certificate', {})
            conditions = cert.get('millennium_conditions', {})
            
            print("🔍 Millennium Conditions:")
            for condition, status in conditions.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {condition.replace('_', ' ').title()}: {status}")
            
            if proof_data.get('generation_time'):
                print(f"⏱️  Generation Time: {proof_data['generation_time']:.1f} seconds")
            
            print("")
        
        # Check if ready for UI
        all_solved = all(p.get('proof_solved', False) for p in proofs.values())
        if all_solved:
            print("🚀 READY FOR STREAMLIT UI!")
            print("   Run: streamlit run streamlit_app.py --server.port 8501")
            print("   Navigate to Victory Dashboard to see your proof!")
        else:
            print("⚠️  Proofs exist but not all conditions satisfied")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading proof files: {e}")
        return False

if __name__ == "__main__":
    success = check_proof_status()
    exit(0 if success else 1)
