package client

import (
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net"
	"strings"
	"time"
)

type TCPClient struct {
	Address string
	Port    string
	conn    net.Conn
}

func NewTCPClient(address, port string) *TCPClient {
	return &TCPClient{
		Address: address,
		Port:    port,
	}
}

func (c *TCPClient) Connect() error {
	var err error
	c.conn, err = net.DialTimeout("tcp", net.JoinHostPort(c.Address, c.Port), 5*time.Second)
	if err != nil {
		return err
	}
	log.Printf("Connected to TCP server at %s:%s", c.Address, c.Port)
	return nil
}

func (c *TCPClient) Disconnect() error {
	if c.conn != nil {
		err := c.conn.Close()
		if err != nil {
			return err
		}
		log.Printf("Disconnected from TCP server at %s:%s", c.Address, c.Port)
	}
	return nil
}

func (c *TCPClient) ReceiveData(buffer []byte) (int, error) {
	if c.conn == nil {
		return 0, net.ErrClosed
	}
	n, err := c.conn.Read(buffer)
	if err != nil {
		return n, err
	}
	return n, nil
}

// ReceiveMessage reads a line-delimited message from the server
func (c *TCPClient) ReceiveMessage() (string, error) {
	if c.conn == nil {
		return "", net.ErrClosed
	}

	buffer := make([]byte, 4096)
	n, err := c.conn.Read(buffer)
	if err != nil {
		return "", err
	}

	// Remove trailing newline if present
	message := string(buffer[:n])
	if len(message) > 0 && message[len(message)-1] == '\n' {
		message = message[:len(message)-1]
	}

	return message, nil
}

func (c *TCPClient) SendData(data []byte) (int, error) {
	if c.conn == nil {
		return 0, net.ErrClosed
	}
	n, err := c.conn.Write(data)
	if err != nil {
		return n, err
	}
	return n, nil
}

// SendMessage sends a string message to the server and waits for acknowledgment
func (c *TCPClient) SendMessage(message string) (string, error) {
	if c.conn == nil {
		return "", net.ErrClosed
	}

	// Append newline for line-delimited protocol
	data := []byte(message + "\n")
	_, err := c.conn.Write(data)
	if err != nil {
		return "", err
	}

	log.Printf("Sent message: %s", message)

	// Wait for acknowledgment
	ack, err := c.ReceiveMessage()
	if err != nil {
		return "", err
	}

	return ack, nil
}

// SendAndReceive sends a message and waits for the full response after acknowledgment.
func (c *TCPClient) SendAndReceive(message string, tokenCallback func(string)) (string, error) {
	ack, err := c.SendMessage(message)
	if err != nil {
		return "", err
	}
	if !strings.EqualFold(ack, "ACK") {
		log.Printf("warning: expected ACK, received %s", ack)
	}

	for {
		line, err := c.ReceiveMessage()
		if err != nil {
			return "", err
		}

		parts := strings.SplitN(line, " ", 2)
		if len(parts) == 2 {
			prefix, payload := parts[0], parts[1]
			if prefix == "TOKN" {
				decoded, err := base64.StdEncoding.DecodeString(payload)
				if err == nil && tokenCallback != nil {
					tokenCallback(string(decoded))
				}
				continue
			}
		}
		return decodeProtocolMessage(line)
	}
}

func decodeProtocolMessage(raw string) (string, error) {
	parts := strings.SplitN(raw, " ", 2)
	if len(parts) != 2 {
		return raw, nil
	}
	prefix, payload := strings.ToUpper(parts[0]), parts[1]
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", fmt.Errorf("failed to decode payload: %w", err)
	}
	switch prefix {
	case "RESP":
		return string(decoded), nil
	case "ERR":
		return "", errors.New(string(decoded))
	default:
		return raw, nil
	}
}
