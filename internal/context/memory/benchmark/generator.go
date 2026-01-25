// Package benchmark provides tools for benchmarking memory systems in OpenEye.
package benchmark

import (
	"fmt"
	"math/rand"
	"time"
)

// Persona represents a synthetic user with consistent attributes.
type Persona struct {
	Name           string
	Occupation     string
	Location       string
	FavoriteColor  string
	Hobbies        []string
	Goals          []string
	FamilyMembers  map[string]string // relation -> name
	ImportantDates map[string]string // event -> date
}

// ConversationTurn represents a single turn in a conversation.
type ConversationTurn struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	// Metadata for validation
	ContainsFact      bool     `json:"contains_fact,omitempty"`
	FactCategories    []string `json:"fact_categories,omitempty"`
	EntitiesMentioned []string `json:"entities_mentioned,omitempty"`
}

// SyntheticConversation is a generated test conversation.
type SyntheticConversation struct {
	ID           string             `json:"id"`
	Persona      Persona            `json:"persona"`
	Turns        []ConversationTurn `json:"turns"`
	PlantedFacts []PlantedFact      `json:"planted_facts"`
}

// PlantedFact is a fact deliberately inserted for recall testing.
type PlantedFact struct {
	TurnIndex      int    `json:"turn_index"`
	Category       string `json:"category"`
	Fact           string `json:"fact"`
	RecallQuery    string `json:"recall_query"`
	ExpectedAnswer string `json:"expected_answer"`
}

// GeneratorConfig configures the conversation generator.
type GeneratorConfig struct {
	NumTurns          int
	TurnIntervalMin   time.Duration
	TurnIntervalMax   time.Duration
	PlantedFactsCount int
	TopicConsistency  float64 // 0.0 = random topics, 1.0 = highly consistent
	Seed              int64
}

// DefaultGeneratorConfig returns sensible defaults.
func DefaultGeneratorConfig() GeneratorConfig {
	return GeneratorConfig{
		NumTurns:          50,
		TurnIntervalMin:   30 * time.Second,
		TurnIntervalMax:   5 * time.Minute,
		PlantedFactsCount: 10,
		TopicConsistency:  0.7,
		Seed:              time.Now().UnixNano(),
	}
}

// Generator creates synthetic conversations for benchmarking.
type Generator struct {
	cfg GeneratorConfig
	rng *rand.Rand
}

// NewGenerator creates a new conversation generator.
func NewGenerator(cfg GeneratorConfig) *Generator {
	return &Generator{
		cfg: cfg,
		rng: rand.New(rand.NewSource(cfg.Seed)),
	}
}

// GeneratePersona creates a random persona with consistent attributes.
func (g *Generator) GeneratePersona() Persona {
	names := []string{"Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Quinn"}
	occupations := []string{"software engineer", "teacher", "designer", "researcher", "nurse", "writer", "chef", "architect"}
	locations := []string{"San Francisco", "New York", "London", "Tokyo", "Berlin", "Sydney", "Toronto", "Singapore"}
	colors := []string{"blue", "green", "purple", "red", "orange", "teal", "black", "yellow"}

	hobbies := []string{"reading", "hiking", "cooking", "gaming", "photography", "painting", "running", "gardening", "music", "travel"}
	goals := []string{
		"learn a new language",
		"get promoted at work",
		"run a marathon",
		"write a book",
		"travel to Japan",
		"learn to play piano",
		"start a side business",
		"improve fitness",
	}

	// Pick random hobbies and goals
	numHobbies := 2 + g.rng.Intn(3)
	numGoals := 1 + g.rng.Intn(3)

	selectedHobbies := g.pickRandom(hobbies, numHobbies)
	selectedGoals := g.pickRandom(goals, numGoals)

	// Generate family
	familyNames := []string{"Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason"}
	family := make(map[string]string)
	if g.rng.Float64() > 0.3 {
		family["spouse"] = familyNames[g.rng.Intn(len(familyNames))]
	}
	if g.rng.Float64() > 0.5 {
		family["sibling"] = familyNames[g.rng.Intn(len(familyNames))]
	}

	// Generate important dates
	dates := make(map[string]string)
	dates["birthday"] = fmt.Sprintf("%s %d", []string{"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}[g.rng.Intn(12)], 1+g.rng.Intn(28))
	if g.rng.Float64() > 0.5 {
		dates["anniversary"] = fmt.Sprintf("%s %d", []string{"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}[g.rng.Intn(12)], 1+g.rng.Intn(28))
	}

	return Persona{
		Name:           names[g.rng.Intn(len(names))],
		Occupation:     occupations[g.rng.Intn(len(occupations))],
		Location:       locations[g.rng.Intn(len(locations))],
		FavoriteColor:  colors[g.rng.Intn(len(colors))],
		Hobbies:        selectedHobbies,
		Goals:          selectedGoals,
		FamilyMembers:  family,
		ImportantDates: dates,
	}
}

func (g *Generator) pickRandom(items []string, n int) []string {
	if n >= len(items) {
		return items
	}
	shuffled := make([]string, len(items))
	copy(shuffled, items)
	g.rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled[:n]
}

// GenerateConversation creates a full synthetic conversation.
func (g *Generator) GenerateConversation() SyntheticConversation {
	persona := g.GeneratePersona()
	conversationID := fmt.Sprintf("conv_%d", time.Now().UnixNano())

	turns := make([]ConversationTurn, 0, g.cfg.NumTurns)
	plantedFacts := make([]PlantedFact, 0, g.cfg.PlantedFactsCount)

	// Decide which turns will have planted facts
	plantedTurnIndices := g.selectPlantedTurns()
	plantedTurnSet := make(map[int]bool)
	for _, idx := range plantedTurnIndices {
		plantedTurnSet[idx] = true
	}

	currentTime := time.Now().Add(-time.Duration(g.cfg.NumTurns) * g.cfg.TurnIntervalMax)
	topics := g.generateTopicSequence(persona)

	for i := 0; i < g.cfg.NumTurns; i++ {
		// Advance time
		diff := int64(g.cfg.TurnIntervalMax - g.cfg.TurnIntervalMin)
		var interval time.Duration
		if diff > 0 {
			interval = g.cfg.TurnIntervalMin + time.Duration(g.rng.Int63n(diff))
		} else {
			interval = g.cfg.TurnIntervalMin
		}
		currentTime = currentTime.Add(interval)

		topic := topics[i%len(topics)]

		// Generate user turn
		userTurn, fact := g.generateUserTurn(persona, topic, i, plantedTurnSet[i])
		userTurn.Timestamp = currentTime
		turns = append(turns, userTurn)

		if fact != nil {
			plantedFacts = append(plantedFacts, *fact)
		}

		// Generate assistant response
		currentTime = currentTime.Add(time.Duration(500+g.rng.Intn(2000)) * time.Millisecond)
		assistantTurn := g.generateAssistantTurn(topic)
		assistantTurn.Timestamp = currentTime
		turns = append(turns, assistantTurn)
	}

	return SyntheticConversation{
		ID:           conversationID,
		Persona:      persona,
		Turns:        turns,
		PlantedFacts: plantedFacts,
	}
}

func (g *Generator) selectPlantedTurns() []int {
	// Distribute planted facts across the conversation
	indices := make([]int, g.cfg.PlantedFactsCount)
	segmentSize := g.cfg.NumTurns / g.cfg.PlantedFactsCount

	for i := 0; i < g.cfg.PlantedFactsCount; i++ {
		start := i * segmentSize
		end := start + segmentSize
		if end > g.cfg.NumTurns {
			end = g.cfg.NumTurns
		}
		indices[i] = start + g.rng.Intn(end-start)
	}
	return indices
}

func (g *Generator) generateTopicSequence(persona Persona) []string {
	topics := []string{
		"work",
		"hobbies",
		"family",
		"goals",
		"daily_life",
		"preferences",
		"memories",
		"planning",
	}

	// Weight topics based on persona
	if len(persona.Hobbies) > 0 {
		topics = append(topics, "hobbies", "hobbies") // More hobby discussion
	}
	if len(persona.Goals) > 0 {
		topics = append(topics, "goals")
	}

	return topics
}

func (g *Generator) generateUserTurn(persona Persona, topic string, turnIndex int, plantFact bool) (ConversationTurn, *PlantedFact) {
	templates := g.getTemplatesForTopic(topic, persona)
	template := templates[g.rng.Intn(len(templates))]

	content := g.fillTemplate(template, persona)

	turn := ConversationTurn{
		Role:         "user",
		Content:      content,
		ContainsFact: plantFact,
	}

	var fact *PlantedFact
	if plantFact {
		fact = g.createPlantedFact(persona, topic, turnIndex, content)
		turn.FactCategories = []string{fact.Category}
	}

	// Extract mentioned entities
	turn.EntitiesMentioned = g.extractEntities(content, persona)

	return turn, fact
}

func (g *Generator) getTemplatesForTopic(topic string, persona Persona) []string {
	switch topic {
	case "work":
		return []string{
			"I had a busy day at work today as a %occupation%.",
			"My project at work is going well. Being a %occupation% can be challenging.",
			"I'm thinking about a new approach to my work as a %occupation%.",
			"Working in %location% has its perks. The %occupation% scene here is great.",
		}
	case "hobbies":
		if len(persona.Hobbies) > 0 {
			return []string{
				"I spent some time %hobby% today and it was relaxing.",
				"I'm getting better at %hobby%. It's become my favorite way to unwind.",
				"Do you have any tips for %hobby%? I want to improve.",
				"I've been into %hobby% for a while now. It's really rewarding.",
			}
		}
		return []string{"I'm looking for a new hobby to try."}
	case "family":
		if len(persona.FamilyMembers) > 0 {
			return []string{
				"I talked to my %family_relation% %family_name% today.",
				"My %family_relation% %family_name% is doing well.",
				"I'm planning to visit %family_name% soon.",
				"I miss spending time with %family_name%.",
			}
		}
		return []string{"I should call my family soon."}
	case "goals":
		if len(persona.Goals) > 0 {
			return []string{
				"I'm working towards my goal to %goal%.",
				"Making progress on my plan to %goal%.",
				"I really want to %goal% this year.",
				"Any advice on how to %goal%?",
			}
		}
		return []string{"I should set some new goals."}
	case "preferences":
		return []string{
			"My favorite color is %color%, by the way.",
			"I really like %color% things.",
			"Living in %location% is great for someone like me.",
			"Being a %occupation% suits my personality.",
		}
	case "daily_life":
		return []string{
			"Just another day here in %location%.",
			"The weather in %location% is nice today.",
			"I grabbed coffee this morning before heading to work.",
			"Trying to maintain a good routine lately.",
		}
	case "memories":
		return []string{
			"I remember when I first moved to %location%.",
			"Back when I started as a %occupation%, things were different.",
			"My birthday is on %birthday%, I always look forward to it.",
			"I've had some great experiences with %hobby%.",
		}
	case "planning":
		return []string{
			"I'm planning something special for %date_event%.",
			"Next week I want to focus on %goal%.",
			"I should schedule time for %hobby%.",
			"Thinking about my plans for the upcoming months.",
		}
	default:
		return []string{"How are you today?", "What do you think about that?", "Tell me more about this topic."}
	}
}

func (g *Generator) fillTemplate(template string, persona Persona) string {
	result := template

	// Replace placeholders
	replacements := map[string]string{
		"%name%":       persona.Name,
		"%occupation%": persona.Occupation,
		"%location%":   persona.Location,
		"%color%":      persona.FavoriteColor,
	}

	if len(persona.Hobbies) > 0 {
		replacements["%hobby%"] = persona.Hobbies[g.rng.Intn(len(persona.Hobbies))]
	}

	if len(persona.Goals) > 0 {
		replacements["%goal%"] = persona.Goals[g.rng.Intn(len(persona.Goals))]
	}

	// Family members
	if len(persona.FamilyMembers) > 0 {
		relations := make([]string, 0, len(persona.FamilyMembers))
		for rel := range persona.FamilyMembers {
			relations = append(relations, rel)
		}
		rel := relations[g.rng.Intn(len(relations))]
		replacements["%family_relation%"] = rel
		replacements["%family_name%"] = persona.FamilyMembers[rel]
	}

	// Dates
	if birthday, ok := persona.ImportantDates["birthday"]; ok {
		replacements["%birthday%"] = birthday
	}
	for event, date := range persona.ImportantDates {
		replacements["%date_event%"] = event + " on " + date
		break
	}

	for placeholder, value := range replacements {
		for {
			newResult := replaceFirst(result, placeholder, value)
			if newResult == result {
				break
			}
			result = newResult
		}
	}

	return result
}

func replaceFirst(s, old, new string) string {
	for i := 0; i <= len(s)-len(old); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + new + s[i+len(old):]
		}
	}
	return s
}

func (g *Generator) createPlantedFact(persona Persona, topic string, turnIndex int, content string) *PlantedFact {
	var category, fact, query, answer string

	switch topic {
	case "preferences":
		category = "preference"
		fact = fmt.Sprintf("User's favorite color is %s", persona.FavoriteColor)
		query = "What is my favorite color?"
		answer = persona.FavoriteColor
	case "work":
		category = "occupation"
		fact = fmt.Sprintf("User works as a %s in %s", persona.Occupation, persona.Location)
		query = "What do I do for work?"
		answer = persona.Occupation
	case "hobbies":
		if len(persona.Hobbies) > 0 {
			hobby := persona.Hobbies[0]
			category = "hobby"
			fact = fmt.Sprintf("User enjoys %s", hobby)
			query = "What are my hobbies?"
			answer = hobby
		}
	case "family":
		if len(persona.FamilyMembers) > 0 {
			for rel, name := range persona.FamilyMembers {
				category = "family"
				fact = fmt.Sprintf("User's %s is named %s", rel, name)
				query = fmt.Sprintf("What is my %s's name?", rel)
				answer = name
				break
			}
		}
	case "goals":
		if len(persona.Goals) > 0 {
			goal := persona.Goals[0]
			category = "goal"
			fact = fmt.Sprintf("User wants to %s", goal)
			query = "What are my goals?"
			answer = goal
		}
	case "memories":
		if birthday, ok := persona.ImportantDates["birthday"]; ok {
			category = "personal_info"
			fact = fmt.Sprintf("User's birthday is on %s", birthday)
			query = "When is my birthday?"
			answer = birthday
		}
	default:
		category = "general"
		fact = fmt.Sprintf("User lives in %s", persona.Location)
		query = "Where do I live?"
		answer = persona.Location
	}

	return &PlantedFact{
		TurnIndex:      turnIndex,
		Category:       category,
		Fact:           fact,
		RecallQuery:    query,
		ExpectedAnswer: answer,
	}
}

func (g *Generator) extractEntities(content string, persona Persona) []string {
	entities := []string{}

	// Check for known entities
	if containsWord(content, persona.Name) {
		entities = append(entities, persona.Name)
	}
	if containsWord(content, persona.Location) {
		entities = append(entities, persona.Location)
	}
	for _, name := range persona.FamilyMembers {
		if containsWord(content, name) {
			entities = append(entities, name)
		}
	}

	return entities
}

func containsWord(s, word string) bool {
	// Simple substring check
	for i := 0; i <= len(s)-len(word); i++ {
		if s[i:i+len(word)] == word {
			return true
		}
	}
	return false
}

func (g *Generator) generateAssistantTurn(topic string) ConversationTurn {
	responses := map[string][]string{
		"work": {
			"That sounds interesting! What aspects of your work do you enjoy most?",
			"Being a professional in your field must come with unique challenges.",
			"It's great that you're engaged with your work.",
		},
		"hobbies": {
			"That's a wonderful hobby! How long have you been doing it?",
			"It's important to have activities that help you relax.",
			"I'd be happy to share some tips if you'd like.",
		},
		"family": {
			"Family connections are so important. How are they doing?",
			"It's nice that you stay in touch with your family.",
			"Spending time with loved ones is valuable.",
		},
		"goals": {
			"That's an admirable goal! What's your next step?",
			"Having clear goals helps maintain focus and motivation.",
			"I can help you break that down into smaller steps if you'd like.",
		},
		"preferences": {
			"That's a nice choice! Colors can say a lot about personality.",
			"I'll remember that preference.",
			"Good to know your tastes.",
		},
		"daily_life": {
			"Sounds like a typical day. Anything special planned?",
			"Routines can be comforting in their own way.",
			"How's the day going so far?",
		},
		"memories": {
			"Memories like that are precious. What made it special?",
			"It's nice to reflect on past experiences.",
			"Those moments shape who we are.",
		},
		"planning": {
			"Planning ahead is always a good idea!",
			"What's the first thing on your list?",
			"I can help you organize those plans.",
		},
	}

	options, ok := responses[topic]
	if !ok {
		options = []string{
			"I understand. Tell me more.",
			"That's interesting. What else is on your mind?",
			"I'm here to help with whatever you need.",
		}
	}

	return ConversationTurn{
		Role:    "assistant",
		Content: options[g.rng.Intn(len(options))],
	}
}

// RecallTest represents a test to check if a fact can be recalled.
type RecallTest struct {
	Query          string `json:"query"`
	ExpectedAnswer string `json:"expected_answer"`
	TurnPlanted    int    `json:"turn_planted"`
	Category       string `json:"category"`
}

// GenerateRecallTests creates recall tests from planted facts.
func (g *Generator) GenerateRecallTests(conv SyntheticConversation) []RecallTest {
	tests := make([]RecallTest, 0, len(conv.PlantedFacts))

	for _, fact := range conv.PlantedFacts {
		tests = append(tests, RecallTest{
			Query:          fact.RecallQuery,
			ExpectedAnswer: fact.ExpectedAnswer,
			TurnPlanted:    fact.TurnIndex,
			Category:       fact.Category,
		})
	}

	return tests
}
