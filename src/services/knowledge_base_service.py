import logging
import json
import hashlib
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeBaseVersioningService:
    """
    Service for managing knowledge base versions and tracking changes.
    """
    
    def __init__(self):
        self.logger = logger
        self.versions = {}
        self.current_version = None
        self.version_history = []
    
    def create_version(self, documents: List[Dict[str, Any]], 
                      version_name: str = None, 
                      description: str = None) -> str:
        """
        Create a new version of the knowledge base.
        
        Args:
            documents: List of documents in the knowledge base
            version_name: Optional name for the version
            description: Optional description of changes
            
        Returns:
            Version identifier
        """
        # Generate version identifier
        timestamp = datetime.now().isoformat()
        if not version_name:
            version_name = f"version_{len(self.versions) + 1}"
        
        # Calculate document hash for change detection
        doc_hash = self._calculate_documents_hash(documents)
        
        # Create version entry
        version_id = f"{version_name}_{timestamp}"
        version_entry = {
            'version_id': version_id,
            'version_name': version_name,
            'description': description,
            'timestamp': timestamp,
            'document_hash': doc_hash,
            'document_count': len(documents),
            'documents': [doc.get('id', f'doc_{i}') for i, doc in enumerate(documents)]
        }
        
        # Store version
        self.versions[version_id] = version_entry
        
        # Add to history
        self.version_history.append(version_entry)
        
        # Set as current version if it's the first one
        if not self.current_version:
            self.current_version = version_id
        
        self.logger.info(f"Created knowledge base version: {version_id}")
        return version_id
    
    def _calculate_documents_hash(self, documents: List[Dict[str, Any]]) -> str:
        """
        Calculate a hash for a set of documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Hash string
        """
        # Create a string representation of all documents
        doc_strings = []
        for doc in documents:
            # Extract key fields for hashing
            doc_id = doc.get('id', '')
            doc_content = doc.get('content', '')
            doc_metadata = str(doc.get('metadata', {}))
            doc_strings.append(f"{doc_id}:{doc_content}:{doc_metadata}")
        
        # Join all document strings
        combined_string = '|'.join(sorted(doc_strings))
        
        # Calculate hash
        return hashlib.sha256(combined_string.encode()).hexdigest()
    
    def get_version(self, version_id: str) -> Dict[str, Any]:
        """
        Get a specific version of the knowledge base.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Version information
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        return self.versions[version_id]
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions of the knowledge base.
        
        Returns:
            List of version information
        """
        return list(self.versions.values())
    
    def compare_versions(self, version1_id: str, 
                        version2_id: str) -> Dict[str, Any]:
        """
        Compare two versions of the knowledge base.
        
        Args:
            version1_id: First version identifier
            version2_id: Second version identifier
            
        Returns:
            Comparison results
        """
        if version1_id not in self.versions:
            raise ValueError(f"Version {version1_id} not found")
        
        if version2_id not in self.versions:
            raise ValueError(f"Version {version2_id} not found")
        
        version1 = self.versions[version1_id]
        version2 = self.versions[version2_id]
        
        # Compare document counts
        doc_count_diff = version2['document_count'] - version1['document_count']
        
        # Compare document lists
        docs1 = set(version1['documents'])
        docs2 = set(version2['documents'])
        
        added_docs = list(docs2 - docs1)
        removed_docs = list(docs1 - docs2)
        common_docs = list(docs1 & docs2)
        
        # Compare document hashes
        hash_changed = version1['document_hash'] != version2['document_hash']
        
        return {
            'version1': version1_id,
            'version2': version2_id,
            'document_count_difference': doc_count_diff,
            'added_documents': added_docs,
            'removed_documents': removed_docs,
            'common_documents': common_docs,
            'hash_changed': hash_changed,
            'timestamp_difference': (
                datetime.fromisoformat(version2['timestamp']) - 
                datetime.fromisoformat(version1['timestamp'])
            ).total_seconds()
        }
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous version of the knowledge base.
        
        Args:
            version_id: Version identifier to rollback to
            
        Returns:
            Success status
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Set as current version
        self.current_version = version_id
        
        self.logger.info(f"Rolled back to knowledge base version: {version_id}")
        return True
    
    def get_current_version(self) -> Dict[str, Any]:
        """
        Get the current version of the knowledge base.
        
        Returns:
            Current version information
        """
        if not self.current_version:
            return None
        
        return self.versions[self.current_version]
    
    def add_document_to_version(self, document: Dict[str, Any], 
                                version_id: str = None) -> bool:
        """
        Add a document to a specific version or current version.
        
        Args:
            document: Document to add
            version_id: Optional version identifier (defaults to current version)
            
        Returns:
            Success status
        """
        if not version_id:
            version_id = self.current_version
        
        # Auto-create initial version if none exists
        if not version_id:
            self.logger.info("No current version set. Creating initial version.")
            version_id = self.create_version(
                documents=[],
                version_name="initial",
                description="Auto-created initial version"
            )
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Add document to version
        version = self.versions[version_id]
        doc_id = document.get('id', f'doc_{len(version["documents"])}')
        version['documents'].append(doc_id)
        version['document_count'] += 1
        
        self.logger.info(f"Added document {doc_id} to version {version_id}")
        return True
    
    def remove_document_from_version(self, document_id: str, 
                                     version_id: str = None) -> bool:
        """
        Remove a document from a specific version or current version.
        
        Args:
            document_id: Document identifier to remove
            version_id: Optional version identifier (defaults to current version)
            
        Returns:
            Success status
        """
        if not version_id:
            version_id = self.current_version
        
        # Auto-create initial version if none exists
        if not version_id:
            self.logger.info("No current version set. Creating initial version.")
            version_id = self.create_version(
                documents=[],
                version_name="initial", 
                description="Auto-created initial version"
            )
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Remove document from version
        version = self.versions[version_id]
        if document_id in version['documents']:
            version['documents'].remove(document_id)
            version['document_count'] -= 1
            
            self.logger.info(f"Removed document {document_id} from version {version_id}")
            return True
        
        self.logger.warning(f"Document {document_id} not found in version {version_id}")
        return False
    
    def export_version(self, version_id: str, filepath: str) -> bool:
        """
        Export a version of the knowledge base to a file.
        
        Args:
            version_id: Version identifier
            filepath: Path to export file
            
        Returns:
            Success status
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        
        try:
            with open(filepath, 'w') as f:
                json.dump(version, f, indent=2)
            
            self.logger.info(f"Exported version {version_id} to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export version {version_id}: {e}")
            return False
    
    def import_version(self, filepath: str) -> str:
        """
        Import a version of the knowledge base from a file.
        
        Args:
            filepath: Path to import file
            
        Returns:
            Version identifier
        """
        try:
            with open(filepath, 'r') as f:
                version_data = json.load(f)
            
            version_id = version_data.get('version_id')
            if not version_id:
                raise ValueError("Invalid version data: missing version_id")
            
            # Store version
            self.versions[version_id] = version_data
            
            self.logger.info(f"Imported version {version_id} from {filepath}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to import version from {filepath}: {e}")
            raise

# Global instance
knowledge_base_service = KnowledgeBaseVersioningService()

def track_document_changes(document: Dict[str, Any],
                          previous_version: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Track changes to a document and generate a change report.
    
    Args:
        document: Current document
        previous_version: Previous version of the document
        
    Returns:
        Change tracking report
    """
    changes = {
        'document_id': document.get('id', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'changes_detected': False,
        'change_type': 'new',
        'content_changes': [],
        'metadata_changes': {}
    }
    
    # If there's no previous version, it's a new document
    if not previous_version:
        changes['changes_detected'] = True
        return changes
    
    # Compare content
    current_content = document.get('content', '')
    previous_content = previous_version.get('content', '')
    
    if current_content != previous_content:
        changes['changes_detected'] = True
        changes['change_type'] = 'content_updated'
        
        # Simple content diff (in a real implementation, you might use a more sophisticated diff algorithm)
        if len(current_content) != len(previous_content):
            changes['content_changes'].append({
                'type': 'length_change',
                'previous_length': len(previous_content),
                'current_length': len(current_content)
            })
    
    # Compare metadata
    current_metadata = document.get('metadata', {})
    previous_metadata = previous_version.get('metadata', {})
    
    metadata_diff = {}
    all_keys = set(current_metadata.keys()) | set(previous_metadata.keys())
    
    for key in all_keys:
        current_value = current_metadata.get(key)
        previous_value = previous_metadata.get(key)
        
        if current_value != previous_value:
            metadata_diff[key] = {
                'previous': previous_value,
                'current': current_value,
                'changed': True
            }
    
    if metadata_diff:
        changes['changes_detected'] = True
        if changes['change_type'] == 'new':
            changes['change_type'] = 'metadata_updated'
        else:
            changes['change_type'] = 'content_and_metadata_updated'
        changes['metadata_changes'] = metadata_diff
    
    return changes