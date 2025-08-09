#!/usr/bin/env python3
"""
Script to clear and recreate the LanceDB table with the correct schema.
"""

import sys
import os
import shutil

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import settings
from src.services.lancedb_service import get_lancedb_service

def clear_lancedb():
    """Clear the existing LanceDB data and recreate the table."""
    
    print("🗑️  Clearing existing LanceDB data...")
    
    # Remove the entire LanceDB directory
    if os.path.exists(settings.LANCEDB_PATH):
        try:
            shutil.rmtree(settings.LANCEDB_PATH)
            print(f"✅ Removed existing LanceDB directory: {settings.LANCEDB_PATH}")
        except Exception as e:
            print(f"❌ Error removing LanceDB directory: {e}")
            return False
    
    # Recreate the LanceDB service and table
    try:
        print("🔄 Creating new LanceDB table with correct schema...")
        lancedb_service = get_lancedb_service()
        lancedb_service.create_table_if_not_exists(
            table_name=settings.LANCEDB_TABLE_NAME,
            force_recreate=True
        )
        
        # Verify the table was created
        stats = lancedb_service.get_table_stats(settings.LANCEDB_TABLE_NAME)
        if stats["exists"]:
            print(f"✅ Successfully created new LanceDB table: {settings.LANCEDB_TABLE_NAME}")
            print(f"📊 Table stats: {stats}")
            return True
        else:
            print("❌ Failed to create LanceDB table")
            return False
            
    except Exception as e:
        print(f"❌ Error creating LanceDB table: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting LanceDB cleanup and recreation...")
    
    success = clear_lancedb()
    
    if success:
        print("\n🎉 LanceDB has been successfully cleared and recreated!")
        print("You can now run your application without schema mismatch errors.")
    else:
        print("\n⚠️  LanceDB cleanup failed. Please check the errors above.")
        sys.exit(1)
