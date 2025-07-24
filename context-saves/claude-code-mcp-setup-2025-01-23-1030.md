# Claude Code MCP Server Setup Session
**Date**: 2025-01-23
**Time**: 10:30 PST
**Focus**: Understanding and enabling MCP servers in Claude Code vs Claude Desktop

## Key Discovery: Claude Code vs Claude Desktop MCP Support

### The Problem
- **Claude Desktop**: Supports all 17 configured MCP servers
- **Claude Code**: Only shows 2 MCP servers (gemini-collab and multi-ai-collab)
- User has zen and graphiti-memory configured and running but not available in Claude Code

### Root Cause
Claude Code has architectural limitations:
1. **Web-based environment** - Cannot access local file systems directly
2. **Security restrictions** - Cannot execute arbitrary local processes
3. **Protocol requirements** - Only supports specific JSON-RPC over stdio pattern
4. **No localhost access** - Cannot connect to Docker containers or local servers

### Why Some MCP Servers Work in Claude Code
The working servers (`gemini-collab` and `multi-ai-collab`) share characteristics:
- Pure Python implementation
- JSON-RPC protocol over stdio
- No local file system dependencies
- No localhost/Docker requirements
- Follow specific naming pattern: `claude_code-*`

## MCP Servers Configured in Claude Desktop

1. **chatgpt-mcp** - ChatGPT integration
2. **mcp-search-linkup** - Linkup search (via uvx)
3. **filesystem** - File system access (npm)
4. **memory** - Memory storage (npm)
5. **brave-search** - Brave search API (npm)
6. **playwright-mcp-server** - Browser automation
7. **puppeteer-mcp-server** - Browser automation
8. **desktop-commander** - Desktop control
9. **github** - GitHub API (npm)
10. **mcp-reddit** - Reddit API
11. **ii-agent** - II Agent with multiple APIs
12. **firecrawl-mcp** - Web scraping (npm)
13. **fin-sales-hybrid** - Financial sales tool
14. **gemini-collab** - Gemini AI (Python) ✅ Works in Claude Code
15. **zen** - Zen MCP server (Python, but not in Claude Code)
16. **multi-ai** - Multi-AI collab (Python) ✅ Works in Claude Code
17. **graphiti-memory** - Knowledge graph (Docker on port 8000)

## Solution: Creating Claude Code Compatible Wrappers

### Template Created
Created `/Users/johnsweazey/claude_code-linkup-mcp/` as a template showing how to wrap MCP servers for Claude Code compatibility.

### Key Implementation Pattern
```python
# 1. Unbuffered stdio
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# 2. JSON-RPC handlers
def handle_initialize(request_id):
def handle_tools_list(request_id):
def handle_tool_call(request_id, params):

# 3. Main loop reading JSON-RPC from stdin
while True:
    line = sys.stdin.readline()
    request = json.loads(line)
    # Route to handlers
    print(json.dumps(response), flush=True)
```

### Files Created
1. `/Users/johnsweazey/claude_code-linkup-mcp/server.py` - Main server implementation
2. `/Users/johnsweazey/claude_code-linkup-mcp/requirements.txt` - Dependencies
3. `/Users/johnsweazey/claude_code-linkup-mcp/setup.sh` - Setup script
4. `/Users/johnsweazey/claude_code-linkup-mcp/README.md` - Documentation

## Important Learnings

### What Works in Claude Code
- Python servers with JSON-RPC over stdio
- API-based services (no local dependencies)
- Simple request/response patterns
- Stateless operations

### What Doesn't Work in Claude Code
- npm-based MCP servers (filesystem, brave-search, etc.)
- Servers requiring local file access
- Docker-based services (graphiti-memory)
- Servers connecting to localhost
- Complex process management

### Why Zen and Graphiti Don't Show Up
1. **Zen** (`/Users/johnsweazey/zen-mcp-server/`):
   - Running as Python process
   - May have dependencies on local file system
   - Might not follow exact Claude Code protocol

2. **Graphiti** (Docker on port 8000):
   - Requires localhost connection
   - Claude Code can't access Docker containers
   - Web-based environment can't reach local ports

## Next Steps for Enabling More MCP Servers in Claude Code

1. **For API-based servers** (like Linkup, Brave Search):
   - Create Python wrappers following the template
   - Use requests library for HTTP calls
   - Implement JSON-RPC protocol

2. **For file-based servers**:
   - Not possible in Claude Code due to security model
   - Must use Claude Desktop

3. **For Docker-based servers** (like Graphiti):
   - Would need cloud hosting with public endpoints
   - Or use Claude Desktop only

## Configuration Addition Template
```json
"your-mcp-code": {
  "command": "python3",
  "args": [
    "/path/to/your/claude_code-your-mcp/server.py"
  ],
  "env": {
    "YOUR_API_KEY": "key-here"
  }
}
```

## Key Takeaway
**Claude Code and Claude Desktop are fundamentally different environments**:
- Claude Desktop: Full local access, all MCP servers work
- Claude Code: Web-based, limited to specific Python implementations

To use zen or graphiti in a conversation, must use Claude Desktop, not Claude Code.

---
*This context save documents the discovery of Claude Code's MCP limitations and provides a template for creating compatible servers.*