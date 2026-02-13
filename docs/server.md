# OpenEye Server

OpenEye provides both HTTP and TCP server modes for programmatic access to the inference pipeline. The HTTP server is the default and recommended option for new integrations.

## Configuration

Server settings are controlled through the `server` section in `openeye.yaml` or environment variables.

### YAML Configuration

```yaml
server:
  type: "http"        # Server type: http or tcp (default: http)
  host: "127.0.0.1"   # Bind address (default: 127.0.0.1)
  port: 8080          # Port number (default: 8080 for HTTP, 42067 for TCP)
  enabled: true       # Enable/disable server
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `APP_SERVER_TYPE` | Server type: `http` or `tcp` |
| `APP_SERVER_HOST` | Override server host address |
| `APP_SERVER_PORT` | Override server port |
| `APP_SERVER_ENABLED` | Enable/disable server (`true` or `false`) |

### CLI Flags

All server settings can be overridden at runtime:

```bash
# Use HTTP server on custom port
OpenEye serve --port 9000

# Use TCP server
OpenEye serve --type tcp --port 42067

# Bind to all interfaces
OpenEye serve --host 0.0.0.0
```

## HTTP Server

The HTTP server provides a REST API with JSON endpoints.

### Endpoints

#### Health Check

```bash
GET /health
```

Returns server status and uptime.

**Response:**
```json
{
  "status": "ok",
  "backend": "http",
  "uptime": "5m30s"
}
```

#### Chat Inference

```bash
POST /v1/chat
```

Send a chat message and receive a response.

**Request Body:**
```json
{
  "message": "Hello, how are you?",
  "images": ["base64encodedimage..."],
  "stream": false,
  "options": {
    "temperature": 0.7,
    "max_tokens": 512
  }
}
```

**Fields:**
- `message` (required): The prompt text
- `images` (optional): Array of base64-encoded images for vision models
- `stream` (optional): Enable streaming response (default: false)
- `options` (optional): Generation parameters
  - `temperature`: Sampling temperature (0.0-2.0)
  - `max_tokens`: Maximum tokens to generate

**Non-Streaming Response:**
```json
{
  "message": "I'm doing well, thank you! How can I help you today?",
  "done": true
}
```

**Streaming Response:**

When `stream: true`, the server returns Server-Sent Events (SSE):

```
Content-Type: text/event-stream

data: {"token": "I'm", "done": false}

data: {"token": " doing", "done": false}

data: {"token": " well", "done": false}

data: {"message": "I'm doing well, thank you!", "done": true}
```

### Image Handling

When using the **native backend** (local llama.cpp), base64 images are automatically:
1. Decoded from base64
2. Saved to temporary files
3. Passed to the native vision pipeline
4. Cleaned up after processing

For the **HTTP backend**, base64 images are passed directly to the configured inference endpoint.

### Examples

#### Simple Chat

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum computing"}'
```

#### Streaming Chat

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a poem", "stream": true}'
```

#### Vision Request

```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 image.jpg)

curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"Describe this image\",
    \"images\": [\"$IMAGE_BASE64\"]
  }"
```

#### With Generation Options

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Creative story idea",
    "options": {
      "temperature": 1.2,
      "max_tokens": 1024
    }
  }'
```

## TCP Server

The TCP server uses a line-based protocol with base64-encoded messages. It's maintained for backward compatibility.

### Protocol

**Request Format:**
```
[IMG:base64data,...] Your message text here\n
```

Or JSON format:
```json
{"images":["base64..."],"content":"Your message"}\n
```

**Response Format:**

Streaming tokens:
```
TOKN <base64 encoded token>\n
```

Final response:
```
RESP <base64 encoded response>\n
```

Error:
```
ERR <base64 encoded error>\n
```

### Examples

#### Simple Request

```bash
echo "Hello, how are you?" | nc localhost 42067
```

#### With Images (Base64)

```bash
echo "[IMG:$(base64 -w 0 image.jpg)] Describe this image" | nc localhost 42067
```

## Comparison

| Feature | HTTP Server | TCP Server |
|---------|-------------|------------|
| Default | Yes | No |
| Port | 8080 | 42067 |
| Protocol | HTTP/REST | TCP/Line-based |
| Streaming | SSE | Custom tokens |
| Health Check | Yes (`/health`) | No |
| JSON | Native | Optional |
| cURL-friendly | Yes | Requires netcat |
| Images | Base64 in JSON | Base64 in payload |

## Migration Guide

### From TCP to HTTP

**TCP Client:**
```go
// Old TCP client
tcp := client.NewTCPClient("127.0.0.1", "42067")
tcp.Connect()
resp, _ := tcp.SendAndReceive("Hello")
```

**HTTP Client:**
```go
// New HTTP client
resp, _ := http.Post("http://127.0.0.1:8080/v1/chat",
    "application/json",
    strings.NewReader(`{"message":"Hello"}`))
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill process or use different port
openeye-native serve --port 9000
```

### Connection Refused

- Verify server is running: `curl http://localhost:8080/health`
- Check firewall settings
- Verify correct host/port in config

### Image Processing Errors

For native backend with vision:
- Ensure images are valid base64
- Supported formats: JPEG, PNG
- Check that mmproj path is configured for multimodal models

## See Also

- [Overview](./overview.md) - Framework architecture
- [Configuration](../openeye.yaml) - Full configuration reference
