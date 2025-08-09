#!/usr/bin/env python3
"""
Simple script to toggle accuracy settings
"""

import os
from src.core.config import settings

def show_current_settings():
    """Show current accuracy settings"""
    print("🎯 Current Accuracy Settings:")
    print("=" * 40)
    print(f"ENABLE_RERANKING: {settings.ENABLE_RERANKING}")
    print(f"USE_ENHANCED_QUERY: {settings.USE_ENHANCED_QUERY}")
    print(f"USE_ENHANCED_RRF: {settings.USE_ENHANCED_RRF}")
    print(f"MAX_CHUNKS_FOR_GENERATION: {settings.MAX_CHUNKS_FOR_GENERATION}")
    print(f"SIMILARITY_THRESHOLD: {settings.SIMILARITY_THRESHOLD}")
    print(f"MAX_GENERATION_TOKENS: {settings.MAX_GENERATION_TOKENS}")
    print(f"GENERATION_TEMPERATURE: {settings.GENERATION_TEMPERATURE}")
    print(f"PARALLEL_PROCESSING: {settings.PARALLEL_PROCESSING}")
    print(f"MAX_PARALLEL_QUESTIONS: {settings.MAX_PARALLEL_QUESTIONS}")
    print("=" * 40)

def toggle_setting(setting_name, new_value):
    """Toggle a setting in the .env file"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print("❌ .env file not found!")
        return
    
    # Read current .env
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update the setting
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{setting_name}="):
            lines[i] = f"{setting_name}={new_value}\n"
            updated = True
            break
    
    # Add setting if not found
    if not updated:
        lines.append(f"{setting_name}={new_value}\n")
    
    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Updated {setting_name}={new_value}")
    print("🔄 Restart your server to apply changes")

def main():
    """Main function"""
    print("🎛️  BajajFinsev Accuracy Control")
    print("=" * 50)
    
    show_current_settings()
    
    print("\nQuick Presets:")
    print("1. Maximum Accuracy (slower)")
    print("2. Balanced (recommended)")
    print("3. Maximum Speed")
    print("4. Custom setting")
    print("5. Show current settings")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        # Maximum accuracy
        toggle_setting("ENABLE_RERANKING", "true")
        toggle_setting("USE_ENHANCED_QUERY", "true")
        toggle_setting("USE_ENHANCED_RRF", "true")
        toggle_setting("MAX_CHUNKS_FOR_GENERATION", "10")
        toggle_setting("SIMILARITY_THRESHOLD", "0.3")
        toggle_setting("GENERATION_TEMPERATURE", "0.01")
        toggle_setting("MAX_PARALLEL_QUESTIONS", "20")  # Slower for accuracy
        print("🎯 Set to Maximum Accuracy mode")
        
    elif choice == "2":
        # Balanced (current defaults)
        toggle_setting("ENABLE_RERANKING", "true")
        toggle_setting("USE_ENHANCED_QUERY", "true")
        toggle_setting("USE_ENHANCED_RRF", "true")
        toggle_setting("MAX_CHUNKS_FOR_GENERATION", "8")
        toggle_setting("SIMILARITY_THRESHOLD", "0.1")
        toggle_setting("GENERATION_TEMPERATURE", "0.05")
        toggle_setting("MAX_PARALLEL_QUESTIONS", "40")
        print("⚖️ Set to Balanced mode")
        
    elif choice == "3":
        # Maximum speed
        toggle_setting("ENABLE_RERANKING", "false")
        toggle_setting("USE_ENHANCED_QUERY", "false")
        toggle_setting("USE_ENHANCED_RRF", "false")
        toggle_setting("MAX_CHUNKS_FOR_GENERATION", "5")
        toggle_setting("SIMILARITY_THRESHOLD", "0.0")  # No threshold for speed
        toggle_setting("GENERATION_TEMPERATURE", "0.1")
        toggle_setting("MAX_PARALLEL_QUESTIONS", "50")  # More parallel for speed
        print("⚡ Set to Maximum Speed mode")
        
    elif choice == "4":
        # Custom setting
        print("\nAvailable settings:")
        print("- ENABLE_RERANKING (true/false)")
        print("- USE_ENHANCED_QUERY (true/false)")
        print("- USE_ENHANCED_RRF (true/false)")
        print("- MAX_CHUNKS_FOR_GENERATION (number)")
        print("- SIMILARITY_THRESHOLD (0.0-1.0)")
        print("- GENERATION_TEMPERATURE (0.01-1.0)")
        print("- MAX_PARALLEL_QUESTIONS (1-50)")
        print("- PARALLEL_PROCESSING (true/false)")
        
        setting = input("Setting name: ").strip()
        value = input("New value: ").strip()
        
        toggle_setting(setting, value)
        
    elif choice == "5":
        show_current_settings()
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
