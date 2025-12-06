package image

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/bmp"
	"golang.org/x/image/draw"
)

// Format represents supported image output formats.
type Format string

const (
	FormatJPEG Format = "jpeg"
	FormatPNG  Format = "png"
	FormatBMP  Format = "bmp"
)

// InputType indicates how the image data is provided.
type InputType string

const (
	InputTypeFilePath InputType = "file"
	InputTypeBase64   InputType = "base64"
	InputTypeRaw      InputType = "raw"
)

// ProcessorConfig defines how images should be processed.
type ProcessorConfig struct {
	// MaxWidth is the maximum width; images wider will be resized.
	MaxWidth int
	// MaxHeight is the maximum height; images taller will be resized.
	MaxHeight int
	// OutputFormat is the target format for processed images.
	OutputFormat Format
	// Quality is the JPEG quality (1-100), only used for JPEG output.
	Quality int
	// PreserveAspectRatio maintains aspect ratio when resizing.
	PreserveAspectRatio bool
	// AutoDetectInput automatically detects if input is base64 or file path.
	AutoDetectInput bool
	// OutputAsBase64 returns processed images as base64 strings.
	OutputAsBase64 bool
}

// DefaultConfig returns a sensible default configuration.
func DefaultConfig() ProcessorConfig {
	return ProcessorConfig{
		MaxWidth:            1024,
		MaxHeight:           1024,
		OutputFormat:        FormatJPEG,
		Quality:             85,
		PreserveAspectRatio: true,
		AutoDetectInput:     true,
		OutputAsBase64:      true,
	}
}

// ProcessedImage holds the result of image processing.
type ProcessedImage struct {
	// Data contains the processed image bytes.
	Data []byte
	// Base64 contains the base64-encoded image (if OutputAsBase64 is true).
	Base64 string
	// Width is the final image width.
	Width int
	// Height is the final image height.
	Height int
	// Format is the output format used.
	Format Format
	// OriginalPath is the source file path (if applicable).
	OriginalPath string
}

// Processor handles image decoding, resizing, and format conversion.
type Processor interface {
	// Process takes an image input and returns a processed image.
	Process(ctx context.Context, input string) (*ProcessedImage, error)
	// ProcessMultiple processes multiple images.
	ProcessMultiple(ctx context.Context, inputs []string) ([]*ProcessedImage, error)
	// Config returns the current configuration.
	Config() ProcessorConfig
}

// DefaultProcessor implements the Processor interface with configurable options.
type DefaultProcessor struct {
	cfg ProcessorConfig
}

// NewProcessor creates a new image processor with the given configuration.
func NewProcessor(cfg ProcessorConfig) *DefaultProcessor {
	if cfg.Quality <= 0 || cfg.Quality > 100 {
		cfg.Quality = 85
	}
	if cfg.MaxWidth <= 0 {
		cfg.MaxWidth = 1024
	}
	if cfg.MaxHeight <= 0 {
		cfg.MaxHeight = 1024
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = FormatJPEG
	}
	return &DefaultProcessor{cfg: cfg}
}

// Config returns the current configuration.
func (p *DefaultProcessor) Config() ProcessorConfig {
	return p.cfg
}

// Process takes an image input (file path or base64) and returns a processed image.
func (p *DefaultProcessor) Process(ctx context.Context, input string) (*ProcessedImage, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	inputType := p.detectInputType(input)
	
	var img image.Image
	var originalPath string
	var err error

	switch inputType {
	case InputTypeFilePath:
		originalPath = input
		img, err = p.loadFromFile(input)
	case InputTypeBase64:
		img, err = p.loadFromBase64(input)
	case InputTypeRaw:
		return nil, fmt.Errorf("raw byte input not supported via string interface")
	default:
		return nil, fmt.Errorf("unknown input type")
	}

	if err != nil {
		return nil, fmt.Errorf("failed to load image: %w", err)
	}

	// Resize if needed
	img = p.resize(img)

	// Encode to target format
	data, err := p.encode(img)
	if err != nil {
		return nil, fmt.Errorf("failed to encode image: %w", err)
	}

	bounds := img.Bounds()
	result := &ProcessedImage{
		Data:         data,
		Width:        bounds.Dx(),
		Height:       bounds.Dy(),
		Format:       p.cfg.OutputFormat,
		OriginalPath: originalPath,
	}

	if p.cfg.OutputAsBase64 {
		result.Base64 = base64.StdEncoding.EncodeToString(data)
	}

	return result, nil
}

// ProcessMultiple processes multiple images concurrently.
func (p *DefaultProcessor) ProcessMultiple(ctx context.Context, inputs []string) ([]*ProcessedImage, error) {
	results := make([]*ProcessedImage, len(inputs))
	errs := make([]error, len(inputs))

	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		results[i], errs[i] = p.Process(ctx, input)
	}

	// Collect errors
	var errMsgs []string
	for i, err := range errs {
		if err != nil {
			errMsgs = append(errMsgs, fmt.Sprintf("image %d: %v", i+1, err))
		}
	}

	if len(errMsgs) > 0 {
		return results, fmt.Errorf("some images failed to process: %s", strings.Join(errMsgs, "; "))
	}

	return results, nil
}

// ProcessBytes processes raw image bytes.
func (p *DefaultProcessor) ProcessBytes(ctx context.Context, data []byte) (*ProcessedImage, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image bytes: %w", err)
	}

	img = p.resize(img)

	encoded, err := p.encode(img)
	if err != nil {
		return nil, fmt.Errorf("failed to encode image: %w", err)
	}

	bounds := img.Bounds()
	result := &ProcessedImage{
		Data:   encoded,
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Format: p.cfg.OutputFormat,
	}

	if p.cfg.OutputAsBase64 {
		result.Base64 = base64.StdEncoding.EncodeToString(encoded)
	}

	return result, nil
}

// detectInputType determines if the input is a file path or base64 data.
func (p *DefaultProcessor) detectInputType(input string) InputType {
	if !p.cfg.AutoDetectInput {
		// Default to file path if auto-detect is disabled
		return InputTypeFilePath
	}

	// Check if it looks like a file path
	if strings.HasPrefix(input, "/") || strings.HasPrefix(input, "./") || 
	   strings.HasPrefix(input, "~") || strings.HasPrefix(input, "../") {
		return InputTypeFilePath
	}

	// Check for Windows-style paths
	if len(input) > 2 && input[1] == ':' {
		return InputTypeFilePath
	}

	// Check if file exists
	if _, err := os.Stat(input); err == nil {
		return InputTypeFilePath
	}

	// Check for data URI scheme
	if strings.HasPrefix(input, "data:image/") {
		return InputTypeBase64
	}

	// Try to detect base64 by checking for valid base64 characters
	// and typical base64 length (longer than typical file paths)
	if len(input) > 100 && isValidBase64(input) {
		return InputTypeBase64
	}

	// Default to file path
	return InputTypeFilePath
}

// isValidBase64 checks if a string contains only valid base64 characters.
func isValidBase64(s string) bool {
	// Remove potential data URI prefix
	if idx := strings.Index(s, ","); idx != -1 {
		s = s[idx+1:]
	}
	
	for _, c := range s {
		if !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || 
		     (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=') {
			return false
		}
	}
	return true
}

// loadFromFile reads an image from a file path.
func (p *DefaultProcessor) loadFromFile(path string) (image.Image, error) {
	// Expand home directory
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, path[2:])
		}
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	return p.decodeImage(file, path)
}

// loadFromBase64 decodes a base64-encoded image.
func (p *DefaultProcessor) loadFromBase64(input string) (image.Image, error) {
	// Handle data URI scheme
	data := input
	if strings.HasPrefix(input, "data:image/") {
		if idx := strings.Index(input, ","); idx != -1 {
			data = input[idx+1:]
		}
	}

	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		// Try URL-safe base64
		decoded, err = base64.URLEncoding.DecodeString(data)
		if err != nil {
			return nil, fmt.Errorf("failed to decode base64: %w", err)
		}
	}

	img, _, err := image.Decode(bytes.NewReader(decoded))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image data: %w", err)
	}

	return img, nil
}

// decodeImage decodes an image from a reader, supporting multiple formats.
func (p *DefaultProcessor) decodeImage(r io.Reader, hint string) (image.Image, error) {
	// Read all data to allow multiple decode attempts
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// Try standard image decode first
	img, format, err := image.Decode(bytes.NewReader(data))
	if err == nil {
		_ = format // Could log or use format info
		return img, nil
	}

	// Try specific formats based on extension
	ext := strings.ToLower(filepath.Ext(hint))
	switch ext {
	case ".bmp":
		return bmp.Decode(bytes.NewReader(data))
	case ".jpg", ".jpeg":
		return jpeg.Decode(bytes.NewReader(data))
	case ".png":
		return png.Decode(bytes.NewReader(data))
	}

	return nil, fmt.Errorf("unsupported image format: %w", err)
}

// resize scales the image to fit within MaxWidth and MaxHeight.
func (p *DefaultProcessor) resize(img image.Image) image.Image {
	bounds := img.Bounds()
	origWidth := bounds.Dx()
	origHeight := bounds.Dy()

	// Check if resizing is needed
	if origWidth <= p.cfg.MaxWidth && origHeight <= p.cfg.MaxHeight {
		return img
	}

	newWidth, newHeight := p.calculateDimensions(origWidth, origHeight)

	// Create resized image
	dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	
	// Use high-quality resampling
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, bounds, draw.Over, nil)

	return dst
}

// calculateDimensions computes new dimensions while respecting constraints.
func (p *DefaultProcessor) calculateDimensions(width, height int) (int, int) {
	if !p.cfg.PreserveAspectRatio {
		newWidth := width
		newHeight := height
		if newWidth > p.cfg.MaxWidth {
			newWidth = p.cfg.MaxWidth
		}
		if newHeight > p.cfg.MaxHeight {
			newHeight = p.cfg.MaxHeight
		}
		return newWidth, newHeight
	}

	// Calculate scaling factor to fit within bounds
	widthRatio := float64(p.cfg.MaxWidth) / float64(width)
	heightRatio := float64(p.cfg.MaxHeight) / float64(height)
	
	ratio := widthRatio
	if heightRatio < widthRatio {
		ratio = heightRatio
	}

	if ratio >= 1.0 {
		return width, height
	}

	return int(float64(width) * ratio), int(float64(height) * ratio)
}

// encode converts the image to the target format.
func (p *DefaultProcessor) encode(img image.Image) ([]byte, error) {
	var buf bytes.Buffer

	switch p.cfg.OutputFormat {
	case FormatJPEG:
		err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: p.cfg.Quality})
		if err != nil {
			return nil, err
		}
	case FormatPNG:
		err := png.Encode(&buf, img)
		if err != nil {
			return nil, err
		}
	case FormatBMP:
		err := bmp.Encode(&buf, img)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported output format: %s", p.cfg.OutputFormat)
	}

	return buf.Bytes(), nil
}

// MimeType returns the MIME type for the configured output format.
func (p *DefaultProcessor) MimeType() string {
	switch p.cfg.OutputFormat {
	case FormatJPEG:
		return "image/jpeg"
	case FormatPNG:
		return "image/png"
	case FormatBMP:
		return "image/bmp"
	default:
		return "application/octet-stream"
	}
}

// DataURI returns a data URI for the processed image.
func (result *ProcessedImage) DataURI() string {
	var mimeType string
	switch result.Format {
	case FormatJPEG:
		mimeType = "image/jpeg"
	case FormatPNG:
		mimeType = "image/png"
	case FormatBMP:
		mimeType = "image/bmp"
	default:
		mimeType = "application/octet-stream"
	}

	b64 := result.Base64
	if b64 == "" {
		b64 = base64.StdEncoding.EncodeToString(result.Data)
	}

	return fmt.Sprintf("data:%s;base64,%s", mimeType, b64)
}
