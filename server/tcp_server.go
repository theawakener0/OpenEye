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
	respond func(responsePayload) error
}

type responsePayload struct {
	content string
	err     error
}

// Respond sends a response back to the originating client.
func (m Message) Respond(response string) error {
	return m.respond(responsePayload{content: response})
}

// RespondError sends an error back to the originating client.
func (m Message) RespondError(err error) error {
	if err == nil {
		return nil
	}
	return m.respond(responsePayload{err: err})
}

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
		message = strings.TrimRight(message, "\r\n")
		log.Printf("Received message: %s", message)

		replyCh := make(chan responsePayload, 1)
		done := make(chan struct{})
		inbound := Message{
			Content: message,
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
		case <-s.shutdown:
			close(done)
			return
		default:
			log.Printf("Message channel full, dropping message")
			if _, err := conn.Write([]byte("ERR " + encodePayload("server busy") + "\n")); err != nil {
				log.Printf("Error notifying client about busy state: %v", err)
			}
			close(done)
			continue
		}

		if _, err := conn.Write([]byte("ACK\n")); err != nil {
			log.Printf("Error sending acknowledgment: %v", err)
			close(done)
			break
		}

		var response responsePayload
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
