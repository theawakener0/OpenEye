# Image Processors

Image processors enable multimodal capabilities in OpenEye, allowing the system to accept and process images alongside text prompts.

## Overview

OpenEye supports vision-capable SLMs that can process images. The image processor handles:

- **Image Loading**: Reading images from files or base64 data
- **Preprocessing**: Resizing, normalizing, format conversion
- **Format Support**: JPEG, PNG, BMP, and other common formats

## Interface Contract

```go
package image

// Processor handles image preprocessing for vision models
type Processor interface {
    // Process converts and normalizes an image
    Process(ctx context.Context, input string) (string, error)
    
    // ProcessBatch processes multiple images
    ProcessBatch(ctx context.Context, inputs []string) ([]string, error)
    
    // Close releases resources
    Close() error
}
```

## Built-in Image Processor

OpenEye's built-in processor handles common preprocessing tasks:

```go
package image

import (
    "bytes"
    "context"
    "encoding/base64"
    "fmt"
    "image"
    "image/jpeg"
    "image/png"
    "os"
    "strings"
)

type Processor struct {
    config ProcessorConfig
}

type ProcessorConfig struct {
    MaxWidth            int
    MaxHeight           int
    OutputFormat        Format
    Quality             int
    PreserveAspectRatio bool
    AutoDetectInput     bool
    OutputAsBase64      bool
}

func NewProcessor(cfg ProcessorConfig) *Processor {
    if cfg.MaxWidth == 0 {
        cfg.MaxWidth = 1024
    }
    if cfg.MaxHeight == 0 {
        cfg.MaxHeight = 1024
    }
    if cfg.Quality == 0 {
        cfg.Quality = 85
    }
    return &Processor{config: cfg}
}

func (p *Processor) Process(ctx context.Context, input string) (string, error) {
    // Detect input type
    if p.config.AutoDetectInput {
        if strings.HasPrefix(input, "data:image") {
            return p.processBase64(ctx, input)
        }
        if _, err := os.Stat(input); err == nil {
            return p.processFile(ctx, input)
        }
    }
    
    // Default: treat as file path
    return p.processFile(ctx, input)
}

func (p *Processor) ProcessBatch(ctx context.Context, inputs []string) ([]string, error) {
    outputs := make([]string, len(inputs))
    
    for i, input := range inputs {
        result, err := p.Process(ctx, input)
        if err != nil {
            return nil, fmt.Errorf("failed to process image %d: %w", i, err)
        }
        outputs[i] = result
    }
    
    return outputs, nil
}

func (p *Processor) processFile(ctx context.Context, path string) (string, error) {
    file, err := os.Open(path)
    if err != nil {
        return "", fmt.Errorf("failed to open image: %w", err)
    }
    defer file.Close()
    
    img, _, err := image.Decode(file)
    if err != nil {
        return "", fmt.Errorf("failed to decode image: %w", err)
    }
    
    return p.processImage(img), nil
}

func (p *Processor) processBase64(ctx context.Context, data string) (string, error) {
    // Extract base64 content
    parts := strings.SplitN(data, ",", 2)
    if len(parts) != 2 {
        return "", fmt.Errorf("invalid base64 image format")
    }
    
    content, err := base64.StdEncoding.DecodeString(parts[1])
    if err != nil {
        return "", fmt.Errorf("failed to decode base64: %w", err)
    }
    
    reader := bytes.NewReader(content)
    img, _, err := image.Decode(reader)
    if err != nil {
        return "", fmt.Errorf("failed to decode image: %w", err)
    }
    
    return p.processImage(img), nil
}

func (p *Processor) processImage(img image.Image) string {
    // Resize if necessary
    resized := p.resizeImage(img)
    
    // Convert to output format
    var buf bytes.Buffer
    
    switch p.config.OutputFormat {
    case FormatJPEG:
        jpeg.Encode(&buf, resized, &jpeg.Options{Quality: p.config.Quality})
    case FormatPNG:
        png.Encode(&buf, resized)
    }
    
    if p.config.OutputAsBase64 {
        return fmt.Sprintf("data:image/%s;base64,%s",
            p.config.OutputFormat,
            base64.StdEncoding.EncodeToString(buf.Bytes()))
    }
    
    return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func (p *Processor) resizeImage(img image.Image) image.Image {
    bounds := img.Bounds()
    width := bounds.Dx()
    height := bounds.Dy()
    
    if width <= p.config.MaxWidth && height <= p.config.MaxHeight {
        return img
    }
    
    // Calculate new size
    var newWidth, newHeight int
    
    if p.config.PreserveAspectRatio {
        ratio := float64(width) / float64(height)
        if width > height {
            newWidth = p.config.MaxWidth
            newHeight = int(float64(newWidth) / ratio)
            if newHeight > p.config.MaxHeight {
                newHeight = p.config.MaxHeight
                newWidth = int(float64(newHeight) * ratio)
            }
        } else {
            newHeight = p.config.MaxHeight
            newWidth = int(float64(newHeight) * ratio)
            if newWidth > p.config.MaxWidth {
                newWidth = p.config.MaxWidth
                newHeight = int(float64(newWidth) / ratio)
            }
        }
    } else {
        newWidth = p.config.MaxWidth
        newHeight = p.config.MaxHeight
    }
    
    // Create resized image
    resized := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
    
    // Simple nearest-neighbor resizing
    for y := 0; y < newHeight; y++ {
        for x := 0; x < newWidth; x++ {
            srcX := x * width / newWidth
            srcY := y * height / newHeight
            resized.Set(x, y, img.At(srcX, srcY))
        }
    }
    
    return resized
}
```

## Custom Image Processor Example

Here's a complete example of a custom image processor with advanced features:

### Project Structure

```
custom-image-processor/
├── go.mod
├── processor.go
└── README.md
```

### processor.go

```go
package customimage

import (
    "bytes"
    "context"
    "encoding/base64"
    "fmt"
    "image"
    "image/color"
    "image/draw"
    "image/jpeg"
    "image/png"
    "math"
    "strings"
    "sync"

    "OpenEye/internal/config"
    "OpenEye/internal/image"
)

const (
    defaultMaxWidth  = 1024
    defaultMaxHeight = 1024
    defaultQuality   = 85
)

type Processor struct {
    config     ProcessorConfig
    cache      *ImageCache
    mu         sync.RWMutex
}

type ProcessorConfig struct {
    MaxWidth            int
    MaxHeight           int
    OutputFormat        image.Format
    Quality             int
    PreserveAspectRatio bool
    Normalize           bool
    TargetMean          float64
    TargetStd           float64
    AutoDetectInput     bool
    OutputAsBase64      bool
}

func NewProcessor(cfg config.ImageConfig) *Processor {
    format := image.FormatJPEG
    switch strings.ToLower(cfg.OutputFormat) {
    case "png":
        format = image.FormatPNG
    case "bmp":
        format = image.FormatBMP
    }
    
    if cfg.MaxWidth == 0 {
        cfg.MaxWidth = defaultMaxWidth
    }
    if cfg.MaxHeight == 0 {
        cfg.MaxHeight = defaultMaxHeight
    }
    if cfg.Quality == 0 {
        cfg.Quality = defaultQuality
    }
    
    return &Processor{
        config: ProcessorConfig{
            MaxWidth:            cfg.MaxWidth,
            MaxHeight:           cfg.MaxHeight,
            OutputFormat:        format,
            Quality:             cfg.Quality,
            PreserveAspectRatio: cfg.PreserveAspectRatio,
            Normalize:           true,
            TargetMean:          0.5,
            TargetStd:           0.5,
            AutoDetectInput:     cfg.AutoDetectInput,
            OutputAsBase64:      cfg.OutputAsBase64,
        },
        cache: NewImageCache(100),
    }
}

func (p *Processor) Process(ctx context.Context, input string) (string, error) {
    // Check cache
    if cached := p.cache.Get(input); cached != "" {
        return cached, nil
    }
    
    var result string
    var err error
    
    if p.config.AutoDetectInput {
        if strings.HasPrefix(input, "data:") {
            result, err = p.processBase64(ctx, input)
        } else if _, err := os.Stat(input); err == nil {
            result, err = p.processFile(ctx, input)
        } else {
            result, err = p.processBase64(ctx, input)
        }
    } else {
        result, err = p.processFile(ctx, input)
    }
    
    if err != nil {
        return "", err
    }
    
    // Cache result
    p.cache.Add(input, result)
    
    return result, nil
}

func (p *Processor) ProcessBatch(ctx context.Context, inputs []string) ([]string, error) {
    results := make([]string, len(inputs))
    var err error
    
    for i, input := range inputs {
        results[i], err = p.Process(ctx, input)
        if err != nil {
            return nil, fmt.Errorf("failed to process image %d: %w", i, err)
        }
    }
    
    return results, nil
}

func (p *Processor) Close() error {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.cache.Clear()
    return nil
}

func (p *Processor) processFile(ctx context.Context, path string) (string, error) {
    file, err := os.Open(path)
    if err != nil {
        return "", fmt.Errorf("failed to open image: %w", err)
    }
    defer file.Close()
    
    img, format, err := image.Decode(file)
    if err != nil {
        return "", fmt.Errorf("failed to decode image: %w", err)
    }
    
    return p.processImage(img, format), nil
}

func (p *Processor) processBase64(ctx context.Context, data string) (string, error) {
    // Extract base64 content
    parts := strings.SplitN(data, ",", 2)
    if len(parts) != 2 {
        return "", fmt.Errorf("invalid base64 image format")
    }
    
    content, err := base64.StdEncoding.DecodeString(parts[1])
    if err != nil {
        return "", fmt.Errorf("failed to decode base64: %w", err)
    }
    
    reader := bytes.NewReader(content)
    img, format, err := image.Decode(reader)
    if err != nil {
        return "", fmt.Errorf("failed to decode image: %w", err)
    }
    
    return p.processImage(img, format), nil
}

func (p *Processor) processImage(img image.Image, format string) string {
    // Resize
    resized := p.resizeImage(img)
    
    // Normalize if enabled
    if p.config.Normalize {
        resized = p.normalizeImage(resized)
    }
    
    // Convert to output format
    return p.encodeImage(resized, format)
}

func (p *Processor) resizeImage(img image.Image) image.Image {
    bounds := img.Bounds()
    width := bounds.Dx()
    height := bounds.Dy()
    
    if width <= p.config.MaxWidth && height <= p.config.MaxHeight {
        return img
    }
    
    var newWidth, newHeight int
    
    if p.config.PreserveAspectRatio {
        ratio := float64(width) / float64(height)
        if width > height {
            newWidth = p.config.MaxWidth
            newHeight = int(float64(newWidth) / ratio)
        } else {
            newHeight = p.config.MaxHeight
            newWidth = int(float64(newHeight) * ratio)
        }
    } else {
        newWidth = p.config.MaxWidth
        newHeight = p.config.MaxHeight
    }
    
    return p.lanczosResize(img, newWidth, newHeight)
}

func (p *Processor) lanczosResize(img image.Image, width, height int) image.Image {
    bounds := img.Bounds()
    srcW := bounds.Dx()
    srcH := bounds.Dy()
    
    resized := image.NewRGBA(image.Rect(0, 0, width, height))
    
    scaleX := float64(srcW) / float64(width)
    scaleY := float64(srcH) / float64(height)
    
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            srcX := float64(x) * scaleX
            srcY := float64(y) * scaleY
            
            color := p.lanczosSample(img, srcX, srcY)
            resized.Set(x, y, color)
        }
    }
    
    return resized
}

func (p *Processor) lanczosSample(img image.Image, x, y float64) color.Color {
    // Simplified Lanczos resampling (2-lobe)
    x0 := int(math.Floor(x - 2))
    x1 := int(math.Floor(x + 2))
    y0 := int(math.Floor(y - 2))
    y1 := int(math.Floor(y + 2))
    
    bounds := img.Bounds()
    
    var r, g, b, a float64
    var total float64
    
    for yy := y0; yy <= y1; yy++ {
        for xx := x0; xx <= x1; xx++ {
            if xx < bounds.Min.X || xx >= bounds.Max.X ||
                yy < bounds.Min.Y || yy >= bounds.Max.Y {
                continue
            }
            
            dx := x - float64(xx)
            dy := y - float64(yy)
            
            weight := p.lanczosKernel(dx) * p.lanczosKernel(dy)
            if weight == 0 {
                continue
            }
            
            c := img.At(xx, yy)
            cr, cg, cb, ca := c.RGBA()
            
            r += float64(cr) * weight
            g += float64(cg) * weight
            b += float64(cb) * weight
            a += float64(ca) * weight
            total += weight
        }
    }
    
    if total == 0 {
        return img.At(int(x), int(y))
    }
    
    return color.RGBA{
        R: uint8(r / total / 257),
        G: uint8(g / total / 257),
        B: uint8(b / total / 257),
        A: uint8(a / total / 257),
    }
}

func (p *Processor) lanczosKernel(x float64) float64 {
    if x == 0 {
        return 1
    }
    if math.Abs(x) >= 2 {
        return 0
    }
    
    piX := math.Pi * x
    return float64(2) * math.Sin(piX) * math.Sin(piX/2) / (piX * piX / 2)
}

func (p *Processor) normalizeImage(img image.Image) image.Image {
    // Calculate mean and std
    mean, std := p.calculateStatistics(img)
    
    // Create normalized image
    normalized := image.NewRGBA(img.Bounds())
    
    scale := p.config.TargetStd / std
    offset := p.config.TargetMean - mean*scale
    
    for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
        for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
            c := img.At(x, y)
            r, g, b, a := c.RGBA()
            
            normalized.Set(x, y, color.RGBA{
                R: uint8(float64(r)*scale + offset),
                G: uint8(float64(g)*scale + offset),
                B: uint8(float64(b)*scale + offset),
                A: uint8(a),
            })
        }
    }
    
    return normalized
}

func (p *Processor) calculateStatistics(img image.Image) (mean, std float64) {
    bounds := img.Bounds()
    total := bounds.Dx() * bounds.Dy()
    var sumR, sumG, sumB float64
    
    for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
        for x := bounds.Min.X; x < bounds.Max.X; x++ {
            c := img.At(x, y)
            r, g, b, _ := c.RGBA()
            sumR += float64(r)
            sumG += float64(g)
            sumB += float64(b)
        }
    }
    
    meanR := sumR / float64(total)
    meanG := sumG / float64(total)
    meanB := sumB / float64(total)
    mean = (meanR + meanG + meanB) / 3 / 257
    
    return mean, std
}

func (p *Processor) encodeImage(img image.Image, format string) string {
    var buf bytes.Buffer
    
    switch p.config.OutputFormat {
    case image.FormatJPEG:
        jpeg.Encode(&buf, img, &jpeg.Options{Quality: p.config.Quality})
    case image.FormatPNG:
        png.Encode(&buf, img)
    }
    
    encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
    
    if p.config.OutputAsBase64 {
        return fmt.Sprintf("data:image/%s;base64,%s", p.config.OutputFormat, encoded)
    }
    
    return encoded
}

// ImageCache for processed images
type ImageCache struct {
    cache map[string]string
    mu    sync.RWMutex
    size  int
}

func NewImageCache(size int) *ImageCache {
    return &ImageCache{
        cache: make(map[string]string),
        size:  size,
    }
}

func (c *ImageCache) Get(key string) string {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.cache[key]
}

func (c *ImageCache) Add(key, value string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if len(c.cache) >= c.size {
        // Simple eviction: clear half
        for k := range c.cache {
            delete(c.cache, k)
            if len(c.cache) < c.size/2 {
                break
            }
        }
    }
    c.cache[key] = value
}

func (c *ImageCache) Clear() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.cache = make(map[string]string)
}
```

## Configuration Integration

### Image Configuration Structure

```go
type ImageConfig struct {
    Enabled             bool   `yaml:"enabled"`
    MaxWidth           int    `yaml:"max_width"`
    MaxHeight          int    `yaml:"max_height"`
    OutputFormat       string `yaml:"output_format"`
    Quality            int    `yaml:"quality"`
    PreserveAspectRatio bool   `yaml:"preserve_aspect_ratio"`
    AutoDetectInput    bool   `yaml:"auto_detect_input"`
    OutputAsBase64     bool   `yaml:"output_as_base64"`
}
```

### YAML Configuration

```yaml
image:
  enabled: true
  max_width: 1024
  max_height: 1024
  output_format: "jpeg"
  quality: 85
  preserve_aspect_ratio: true
  auto_detect_input: true
  output_as_base64: true
```

## Security Considerations

### 1. Path Traversal Prevention

```go
func (p *Processor) processFile(ctx context.Context, path string) (string, error) {
    // Resolve to absolute path
    absPath, err := filepath.Abs(path)
    if err != nil {
        return "", fmt.Errorf("invalid path: %w", err)
    }
    
    // Ensure path is within allowed directory
    allowedDir := p.config.BaseDirectory
    if !strings.HasPrefix(absPath, allowedDir) {
        return "", errors.New("path outside allowed directory")
    }
    
    // ... continue
}
```

### 2. File Type Validation

```go
func (p *Processor) validateImage(path string) error {
    // Check file extension
    ext := strings.ToLower(filepath.Ext(path))
    allowed := map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".bmp": true}
    if !allowed[ext] {
        return fmt.Errorf("unsupported image format: %s", ext)
    }
    
    // Check file header (magic numbers)
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    header := make([]byte, 8)
    _, err = file.Read(header)
    if err != nil {
        return err
    }
    
    // Verify magic numbers
    if !isValidImageHeader(header) {
        return errors.New("invalid image file header")
    }
    
    return nil
}

func isValidImageHeader(header []byte) bool {
    jpegHeader := []byte{0xFF, 0xD8, 0xFF}
    pngHeader := []byte{0x89, 0x50, 0x4E, 0x47}
    bmpHeader := []byte{0x42, 0x4D}
    
    return bytes.HasPrefix(header, jpegHeader) ||
           bytes.HasPrefix(header, pngHeader) ||
           bytes.HasPrefix(header, bmpHeader)
}
```

### 3. File Size Limits

```go
func (p *Processor) validateFileSize(path string) error {
    info, err := os.Stat(path)
    if err != nil {
        return err
    }
    
    maxSize := 10 * 1024 * 1024 // 10MB
    if info.Size() > int64(maxSize) {
        return fmt.Errorf("image exceeds maximum size of %d bytes", maxSize)
    }
    
    return nil
}
```

## Performance Optimization

### 1. Parallel Processing

```go
func (p *Processor) ProcessBatch(ctx context.Context, inputs []string) ([]string, error) {
    results := make([]string, len(inputs))
    errors := make(chan error, len(inputs))
    
    var wg sync.WaitGroup
    
    for i, input := range inputs {
        wg.Add(1)
        go func(idx int, imgPath string) {
            defer wg.Done()
            
            result, err := p.Process(ctx, imgPath)
            if err != nil {
                errors <- fmt.Errorf("image %d: %w", idx, err)
                return
            }
            results[idx] = result
        }(i, input)
    }
    
    wg.Wait()
    close(errors)
    
    for err := range errors {
        return nil, err
    }
    
    return results, nil
}
```

### 2. Caching

```go
func (p *Processor) Process(ctx context.Context, input string) (string, error) {
    // Check cache first
    if cached := p.cache.Get(input); cached != "" {
        return cached, nil
    }
    
    result, err := p.processImage(input)
    if err != nil {
        return "", err
    }
    
    // Cache result
    p.cache.Add(input, result)
    
    return result, nil
}
```

## Testing Image Processors

### Unit Testing

```go
func TestProcessor_Process(t *testing.T) {
    processor := NewProcessor(config.ImageConfig{
        MaxWidth:            1024,
        MaxHeight:           1024,
        OutputFormat:        "jpeg",
        Quality:             85,
        PreserveAspectRatio: true,
        AutoDetectInput:     true,
        OutputAsBase64:      true,
    })
    
    // Test with valid image
    result, err := processor.Process(context.Background(), "testdata/test.jpg")
    if err != nil {
        t.Fatalf("Process failed: %v", err)
    }
    
    if result == "" {
        t.Error("Expected non-empty result")
    }
    
    if !strings.HasPrefix(result, "data:image/jpeg;base64,") {
        t.Error("Expected base64 JPEG output")
    }
}

func TestProcessor_InvalidFormat(t *testing.T) {
    processor := NewProcessor(config.ImageConfig{})
    
    _, err := processor.Process(context.Background(), "testdata/invalid.xyz")
    if err == nil {
        t.Error("Expected error for invalid format")
    }
}
```

## Common Issues

### Memory Usage

```go
func (p *Processor) processLargeImage(path string) error {
    // Use streaming to avoid loading entire image into memory
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    // Decode incrementally
    decoder := png.NewDecoder(file)
    bounds := decoder.Bounds()
    
    // Process in chunks
    chunkSize := 256
    for y := bounds.Min.Y; y < bounds.Max.Y; y += chunkSize {
        img, err := decoder.DecodeImage(bounds.Min.X, min(bounds.Max.Y, y+chunkSize))
        if err != nil {
            return err
        }
        
        // Process chunk
        processChunk(img)
    }
    
    return nil
}
```

### Aspect Ratio Distortion

```go
func (p *Processor) resizeImage(img image.Image) image.Image {
    bounds := img.Bounds()
    width := bounds.Dx()
    height := bounds.Dy()
    
    // Calculate ratio
    ratio := float64(width) / float64(height)
    
    var newWidth, newHeight int
    
    if width > p.config.MaxWidth {
        newWidth = p.config.MaxWidth
        newHeight = int(float64(newWidth) / ratio)
    }
    
    if newHeight > p.config.MaxHeight {
        newHeight = p.config.MaxHeight
        newWidth = int(float64(newHeight) * ratio)
    }
    
    // Use high-quality resizing
    return p.lanczosResize(img, newWidth, newHeight)
}
```

## Checklist for Production

- [ ] Supported formats implemented
- [ ] Resizing with aspect ratio preservation
- [ ] Path traversal prevention
- [ ] File type validation
- [ ] File size limits
- [ ] Memory-efficient processing
- [ ] Caching implemented
- [ ] Error handling complete
- [ ] Unit tests passing
- [ ] Integration tests passing

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [Best Practices](best-practices.md)
