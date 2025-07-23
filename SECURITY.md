# Security Policy

## Overview

Instant-DB is designed with privacy and security as core principles. All processing happens locally on your machine, ensuring your documents never leave your control.

## Privacy Guarantees

### 🔒 100% Local Processing
- **No cloud services**: All document processing, embedding generation, and searches happen on your local machine
- **No external API calls**: Unless you explicitly configure external providers (like OpenAI), Instant-DB operates completely offline
- **No telemetry**: We don't collect any usage data, analytics, or telemetry
- **No phone home**: Instant-DB never connects to our servers or any third-party services without your explicit configuration

### 📁 Data Storage
- **Local database only**: All data is stored in local SQLite and vector databases in your project directory
- **Portable data**: Your `instant_db_data/` directory contains all your data and can be moved, backed up, or deleted at will
- **No hidden caches**: All data storage locations are transparent and documented

### 🔐 Data Security
- **File access**: Instant-DB only reads files you explicitly point it to
- **No network access**: By default, Instant-DB has no network capabilities
- **Embedding privacy**: When using local models (default), your document contents never leave your machine
- **API key security**: If you choose to use external providers, API keys are stored in environment variables, never in code

## GDPR Compliance

Instant-DB is designed to be GDPR-compliant by default:

### Data Minimization
- Only processes documents you explicitly provide
- Only extracts and stores necessary information for search functionality
- No personal data collection beyond document contents

### Right to Erasure
- Delete your `instant_db_data/` directory to remove all processed data
- Use `instant-db clear` to remove specific documents or all data
- No hidden backups or caches remain after deletion

### Data Portability
- All data is stored in standard formats (SQLite, JSON, vector indices)
- Export functionality available via `instant-db export`
- Your data can be moved between machines freely

### Transparency
- Open source codebase - inspect exactly what happens to your data
- Clear documentation of all data processing steps
- No hidden functionality or undocumented features

## Security Best Practices

### For Users

1. **File System Security**
   - Ensure your `instant_db_data/` directory has appropriate permissions
   - Consider encrypting your disk if processing sensitive documents
   - Regularly backup your data directory

2. **API Key Management** (if using external providers)
   - Never commit `.env` files to version control
   - Use environment variables for API keys
   - Rotate API keys regularly
   - Use read-only API keys where possible

3. **Network Security**
   - Run Instant-DB on trusted networks only
   - If using the REST API server, implement authentication
   - Use firewall rules to restrict access to the API port

### For Developers

1. **Code Security**
   - All dependencies are pinned to specific versions
   - Regular security updates for dependencies
   - No execution of arbitrary code from documents
   - Input sanitization for all user inputs

2. **Secure Defaults**
   - Local processing is the default
   - No network access without explicit configuration
   - Conservative file permissions on created files
   - Secure random number generation for IDs

## Vulnerability Reporting

We take security seriously. If you discover a security vulnerability:

1. **Do NOT** create a public GitHub issue
2. Email security concerns to: [maintainer email - update this]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide updates on the fix.

## Security Features

### Document Processing
- **Sandboxed extraction**: Document parsing doesn't execute macros or scripts
- **Type validation**: All file types are validated before processing
- **Size limits**: Configurable limits prevent resource exhaustion
- **Safe parsing**: Using well-tested libraries with security track records

### Database Security
- **SQL injection prevention**: All queries use parameterized statements
- **Access control**: Database files created with restrictive permissions
- **Integrity checks**: Vector indices include checksums
- **Atomic operations**: Database corruption prevention

### Memory Safety
- **Python memory management**: Automatic garbage collection
- **Resource limits**: Configurable memory limits for large documents
- **Streaming processing**: Large files processed in chunks
- **Clean shutdown**: Proper resource cleanup on exit

## Third-Party Dependencies

We carefully vet all dependencies for security:

- **ChromaDB**: Local vector storage, no external connections
- **FAISS**: Facebook's vector similarity search, runs locally
- **SQLite**: Battle-tested embedded database
- **Sentence-transformers**: Can run fully offline with local models
- **Python-magic**: File type detection, no code execution

## Compliance Certifications

While Instant-DB itself is not certified, it is designed to be compatible with:

- ✅ GDPR (General Data Protection Regulation)
- ✅ CCPA (California Consumer Privacy Act)
- ✅ HIPAA (when properly configured and secured)
- ✅ SOC 2 (within properly controlled environments)

## Security Checklist

Before processing sensitive documents:

- [ ] Running on a secure, encrypted machine
- [ ] File system permissions properly configured
- [ ] No untrusted users have access to the system
- [ ] API keys (if any) stored securely
- [ ] Regular backups configured
- [ ] Monitoring disk space for vector databases
- [ ] Updated to latest Instant-DB version

## Updates and Patches

- Security updates are released as soon as vulnerabilities are discovered
- Update notifications in CLI when new versions are available
- Changelogs clearly mark security fixes
- Subscribe to GitHub releases for security notifications

## Contact

For security concerns, questions about privacy, or compliance inquiries:
- GitHub Issues: https://github.com/MakeWorkTakesWork/Instant-DB/issues (for non-sensitive issues)
- Security Email: [maintainer email - update this]
- Documentation: https://github.com/MakeWorkTakesWork/Instant-DB/wiki

---

*Last updated: January 2025*
*Instant-DB Version: 1.0+*