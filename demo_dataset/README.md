# 📚 Instant-DB Demo Dataset

This demo dataset contains sample documents to help you quickly test Instant-DB's capabilities.

## 📁 Contents

1. **sample_pitch_deck.txt** - A sample SaaS pitch deck content
2. **objection_handling.txt** - Common sales objections and responses
3. **product_features.txt** - Product feature documentation
4. **meeting_notes.txt** - Sample meeting notes with action items

## 🚀 Quick Start

```bash
# Process the demo dataset
instant-db process ./demo_dataset

# Try some searches
instant-db search "pricing objections"
instant-db search "integration capabilities"
instant-db search "customer success metrics"

# Search with graph relationships
instant-db search "who mentioned enterprise pricing" --graph

# Export results
instant-db export --format markdown
```

## 💡 What to Look For

1. **Entity Extraction**: Notice how people, companies, and products are identified
2. **Relationship Mapping**: See connections between entities across documents
3. **Context Understanding**: Search for concepts not explicitly mentioned
4. **Metadata Filtering**: Try filtering by document type or date

## 🎯 Sample Queries to Try

- "What are the main objections to our pricing?"
- "Which features do enterprise customers care about?"
- "Who are the key stakeholders mentioned?"
- "What integrations do we support?"
- "Show me all action items from meetings"

## 📊 Expected Results

You should see:
- Relevant chunks from multiple documents
- Entity relationships visualization (if using --graph)
- Confidence scores for each result
- Source document references