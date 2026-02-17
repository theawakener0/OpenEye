package memory

import (
	"sync"
	"time"
)

type SessionTurn struct {
	Role    string
	Content string
	Time    time.Time
}

type SessionCache struct {
	mu       sync.RWMutex
	turns    []SessionTurn
	maxTurns int
	loaded   bool
}

func NewSessionCache(maxTurns int) *SessionCache {
	if maxTurns <= 0 {
		maxTurns = 20
	}
	return &SessionCache{
		turns:    make([]SessionTurn, 0, maxTurns),
		maxTurns: maxTurns,
		loaded:   false,
	}
}

func (sc *SessionCache) AddTurn(role, content string) {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	sc.turns = append(sc.turns, SessionTurn{
		Role:    role,
		Content: content,
		Time:    time.Now(),
	})

	if len(sc.turns) > sc.maxTurns {
		sc.turns = sc.turns[len(sc.turns)-sc.maxTurns:]
	}
}

func (sc *SessionCache) GetRecentTurns(n int) []SessionTurn {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if n <= 0 || n > len(sc.turns) {
		n = len(sc.turns)
	}
	result := make([]SessionTurn, n)
	copy(result, sc.turns[len(sc.turns)-n:])
	return result
}

func (sc *SessionCache) GetAllTurns() []SessionTurn {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	result := make([]SessionTurn, len(sc.turns))
	copy(result, sc.turns)
	return result
}

func (sc *SessionCache) Clear() {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.turns = sc.turns[:0]
	sc.loaded = false
}

func (sc *SessionCache) Len() int {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return len(sc.turns)
}

func (sc *SessionCache) IsLoaded() bool {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.loaded
}

func (sc *SessionCache) SetLoaded() {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.loaded = true
}
