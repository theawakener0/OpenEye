package server

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net"
	"strings"
	"sync"
)

type TCPServer struct {
	Address        string
	Port           string
	ln             net.Listener
	messageChannel chan Message
	mu             sync.RWMutex
	shutdown       chan struct{}
}

// Message represents a single inbound payload with facilities to respond.
type Message struct {
	Content string
	Images  []string // Base64-encoded images or file paths
	respond func(responsePayload) error
	stream  func(string) error
}

type responsePayload struct {
	content string
	err     error
}

// Respond sends a response back to the originating client.
func (m Message) Respond(response string) error {
	return m.respond(responsePayload{content: response})
}

// StreamToken sends a partial token back to the originating client.
func (m Message) StreamToken(token string) error {
	if m.stream != nil {
		return m.stream(token)
	}
	return nil
}

// RespondError sends an error back to the originating client.
func (m Message) RespondError(err error) error {
	if err == nil {
		return nil
	}
	return m.respond(responsePayload{err: err})
}

// NewTCPServer creates a new TCP server instance.
func NewTCPServer(address, port string) *TCPServer {
	return &TCPServer{
		Address:        address,
		Port:           port,
		messageChannel: make(chan Message, 100),
		shutdown:       make(chan struct{}),
	}
}

func (s *TCPServer) Start() error {
	var err error
	s.ln, err = net.Listen("tcp", net.JoinHostPort(s.Address, s.Port))
	if err != nil {
		return err
	}
	log.Printf("TCP server started on %s:%s", s.Address, s.Port)

	go func() {
		for {
			conn, err := s.ln.Accept()
			if err != nil {
				// Check if server is shutting down
				select {
				case <-s.shutdown:
					return
				default:
					log.Printf("Error accepting connection: %v", err)
					return
				}
			}
			go s.handleConnection(conn)
		}
	}()

	return nil
}

func (s *TCPServer) handleConnection(conn net.Conn) {
	defer conn.Close()
	log.Printf("New connection from %s", conn.RemoteAddr().String())

	reader := bufio.NewReader(conn)
	for {
		message, err := reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				log.Printf("Error reading from connection: %v", err)
			}
			break
		}

		if len(message) == 0 {
			continue
		}
		streamCh := make(chan string, 100)
		message = strings.TrimRight(message, "\r\n")

		// Parse message format: [IMG:base64,base64,...] content
		// or just: content (no images)
		content, images := parseMessageWithImages(message)
		log.Printf("Received message: %s (images: %d)", truncateLog(content, 100), len(images))

		replyCh := make(chan responsePayload, 1)
		done := make(chan struct{})
		var response responsePayload

		inbound := Message{
			stream: func(token string) error {
				select {
				case <-done:
					return fmt.Errorf("connection closed")
				case streamCh <- token:
					return nil
				}
			},
			Content: content,
			Images:  images,
			respond: func(resp responsePayload) error {
				select {
				case <-done:
					return fmt.Errorf("connection closed")
				case replyCh <- resp:
					return nil
				}
			},
		}

		select {
		case s.messageChannel <- inbound:

			// Response loop: handle both streaming tokens and the final response
		Loop:
			for {
				select {
				case token := <-streamCh:
					line := "TOKN " + encodePayload(token) + "\n"
					if _, err := conn.Write([]byte(line)); err != nil {
						log.Printf("Error sending token: %v", err)
						close(done)
						break Loop
					}
				case res := <-replyCh:
					response = res
					break Loop
				case <-s.shutdown:
					close(done)
					return
				}
			}

			// If done is already closed (error case), return
			select {
			case <-done:
				return
			default:
				close(done)
			}
			close(done)
			continue
		}

		if _, err := conn.Write([]byte("ACK\n")); err != nil {
			log.Printf("Error sending acknowledgment: %v", err)
			close(done)
			break
		}

		select {
		case response = <-replyCh:
		case <-s.shutdown:
			close(done)
			return
		}
		close(done)

		var line string
		if response.err != nil {
			line = "ERR " + encodePayload(response.err.Error()) + "\n"
		} else {
			line = "RESP " + encodePayload(response.content) + "\n"
		}

		if _, err := conn.Write([]byte(line)); err != nil {
			log.Printf("Error sending response: %v", err)
			break
		}
	}
	log.Printf("Connection from %s closed", conn.RemoteAddr().String())
}

func (s *TCPServer) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	close(s.shutdown)

	if s.ln != nil {
		err := s.ln.Close()
		if err != nil {
			return err
		}
		log.Printf("TCP server on %s:%s stopped", s.Address, s.Port)
	}

	close(s.messageChannel)
	return nil
}

func (s *TCPServer) GetAddress() string {
	return s.Address
}

func (s *TCPServer) GetPort() string {
	return s.Port
}
func (s *TCPServer) GetListener() net.Listener {
	return s.ln
}

func (s *TCPServer) IsRunning() bool {
	return s.ln != nil
}
func (s *TCPServer) SetAddress(address string) {
	s.Address = address
}
func (s *TCPServer) SetPort(port string) {
	s.Port = port
}
func (s *TCPServer) SetListener(ln net.Listener) {
	s.ln = ln
}

// ReceiveMessage retrieves the next message from the message channel
// Returns empty string and error if channel is closed or timeout occurs
func (s *TCPServer) Receive() (Message, error) {
	select {
	case msg, ok := <-s.messageChannel:
		if !ok {
			return Message{}, fmt.Errorf("message channel closed")
		}
		return msg, nil
	case <-s.shutdown:
		return Message{}, fmt.Errorf("server is shutting down")
	}
}

func (s *TCPServer) ReceiveMessage() (string, error) {
	msg, err := s.Receive()
	if err != nil {
		return "", err
	}
	return msg.Content, nil
}

// ReceiveData reads data from a connection into a buffer (low-level method)
func (s *TCPServer) ReceiveData(conn net.Conn, buffer []byte) (int, error) {
	return conn.Read(buffer)
}

// SendData writes data to a connection (low-level method)
func (s *TCPServer) SendData(conn net.Conn, data []byte) (int, error) {
	return conn.Write(data)
}

func encodePayload(payload string) string {
	return base64.StdEncoding.EncodeToString([]byte(payload))
}

func parseMessageWithImages(raw string) (content string, images []string) {
	content = raw

	// Check for [IMG:...] prefix format (comma-separated in single block)
	if strings.HasPrefix(raw, "[IMG:") {
		endIdx := strings.Index(raw, "]")
		if endIdx > 5 {
			imgData := raw[5:endIdx]
			// Split by comma for multiple images
			for _, img := range strings.Split(imgData, ",") {
				img = strings.TrimSpace(img)
				if img != "" {
					images = append(images, img)
				}
			}
			content = strings.TrimSpace(raw[endIdx+1:])

			// Check for additional [IMG:...] blocks
			for strings.HasPrefix(content, "[IMG:") {
				nextEnd := strings.Index(content, "]")
				if nextEnd > 5 {
					imgData := content[5:nextEnd]
					for _, img := range strings.Split(imgData, ",") {
						img = strings.TrimSpace(img)
						if img != "" {
							images = append(images, img)
						}
					}
					content = strings.TrimSpace(content[nextEnd+1:])
				} else {
					break
				}
			}
			return content, images
		}
	}

	// Check for JSON-like format: {"images":["...","..."],"content":"..."}
	if strings.HasPrefix(raw, "{") && strings.Contains(raw, "\"images\"") {
		content, images = parseJSONMessage(raw)
		if content != "" || len(images) > 0 {
			return content, images
		}
	}

	return raw, nil
}

// parseJSONMessage attempts to parse a simple JSON format for messages with images.
// Format: {"images":["base64_1","base64_2"],"content":"message text"}
func parseJSONMessage(raw string) (content string, images []string) {
	// Simple parsing without full JSON library for performance
	// Extract content field
	contentStart := strings.Index(raw, "\"content\"")
	if contentStart != -1 {
		// Find the value after "content":
		afterContent := raw[contentStart+9:]
		colonIdx := strings.Index(afterContent, ":")
		if colonIdx != -1 {
			afterColon := strings.TrimSpace(afterContent[colonIdx+1:])
			if strings.HasPrefix(afterColon, "\"") {
				// Find the closing quote (handle escaped quotes)
				content = extractQuotedString(afterColon)
			}
		}
	}

	// Extract images array
	imagesStart := strings.Index(raw, "\"images\"")
	if imagesStart != -1 {
		afterImages := raw[imagesStart+8:]
		colonIdx := strings.Index(afterImages, ":")
		if colonIdx != -1 {
			afterColon := strings.TrimSpace(afterImages[colonIdx+1:])
			if strings.HasPrefix(afterColon, "[") {
				// Find closing bracket
				bracketEnd := findMatchingBracket(afterColon)
				if bracketEnd > 0 {
					arrayContent := afterColon[1:bracketEnd]
					// Extract each quoted string
					for len(arrayContent) > 0 {
						arrayContent = strings.TrimSpace(arrayContent)
						if strings.HasPrefix(arrayContent, "\"") {
							img := extractQuotedString(arrayContent)
							if img != "" {
								images = append(images, img)
							}
							// Move past this string and comma
							endQuote := strings.Index(arrayContent[1:], "\"")
							if endQuote != -1 {
								arrayContent = arrayContent[endQuote+2:]
								commaIdx := strings.Index(arrayContent, ",")
								if commaIdx != -1 {
									arrayContent = arrayContent[commaIdx+1:]
								} else {
									break
								}
							} else {
								break
							}
						} else {
							break
						}
					}
				}
			}
		}
	}

	return content, images
}

// extractQuotedString extracts a quoted string value, handling basic escapes.
func extractQuotedString(s string) string {
	if !strings.HasPrefix(s, "\"") {
		return ""
	}
	s = s[1:] // Skip opening quote
	var result strings.Builder
	escaped := false
	for _, c := range s {
		if escaped {
			switch c {
			case 'n':
				result.WriteRune('\n')
			case 't':
				result.WriteRune('\t')
			case 'r':
				result.WriteRune('\r')
			default:
				result.WriteRune(c)
			}
			escaped = false
		} else if c == '\\' {
			escaped = true
		} else if c == '"' {
			return result.String()
		} else {
			result.WriteRune(c)
		}
	}
	return result.String()
}

// findMatchingBracket finds the index of the closing bracket.
func findMatchingBracket(s string) int {
	if !strings.HasPrefix(s, "[") {
		return -1
	}
	depth := 0
	inQuote := false
	escaped := false
	for i, c := range s {
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == '"' {
			inQuote = !inQuote
			continue
		}
		if inQuote {
			continue
		}
		if c == '[' {
			depth++
		} else if c == ']' {
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// truncateLog truncates a string for logging.
func truncateLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
