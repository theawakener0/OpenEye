package omem

import (
	"fmt"
	"strings"
)

// PromptTemplates contains all prompts used by the Omem system.
// These prompts are optimized for small language models (SLMs) with:
// - Clear, structured output formats using line-based parsing (not JSON)
// - Explicit field names and delimiters for reliability
// - Short, focused instructions for limited context windows
// - Examples only when essential for disambiguation
type PromptTemplates struct{}

// NewPromptTemplates creates a new prompt template provider.
func NewPromptTemplates() *PromptTemplates {
	return &PromptTemplates{}
}

// ConversationTurn represents a single turn in a conversation.
type ConversationTurn struct {
	Role    string // "user" or "assistant"
	Content string
	TurnID  string // Optional identifier for the turn
}

// ============================================================================
// Atomic Encoding Prompts
// ============================================================================

// CoreferenceResolutionPrompt generates a prompt for resolving pronouns.
// This is used when rule-based coreference fails and LLM help is needed.
func (p *PromptTemplates) CoreferenceResolutionPrompt(text string, knownEntities []string) string {
	entityList := ""
	if len(knownEntities) > 0 {
		entityList = "KNOWN ENTITIES: " + strings.Join(knownEntities, ", ") + "\n\n"
	}

	return fmt.Sprintf(`Resolve pronouns to their actual entity names.

%sINPUT: %s

OUTPUT the same text with pronouns replaced by entity names.
Only replace pronouns (he/she/it/they/his/her/their/them) that clearly refer to known entities.
Do not change anything else.

RESOLVED TEXT:`, entityList, text)
}

// TemporalAnchoringPrompt generates a prompt for resolving relative time references.
// This is used when rule-based temporal anchoring fails.
func (p *PromptTemplates) TemporalAnchoringPrompt(text string, currentDate string) string {
	return fmt.Sprintf(`Convert relative time references to absolute dates.

CURRENT DATE: %s
INPUT: %s

OUTPUT the same text with time references converted.
Convert: yesterday, today, tomorrow, last week, next month, etc.
Format dates as YYYY-MM-DD.

ANCHORED TEXT:`, currentDate, text)
}

// ============================================================================
// Fact Extraction Prompts
// ============================================================================

// AtomicFactExtractionPrompt generates a prompt for extracting atomic facts.
// Optimized for SLMs: uses simple line-based format instead of JSON.
func (p *PromptTemplates) AtomicFactExtractionPrompt(preprocessedText string, maxFacts int) string {
	return fmt.Sprintf(`Extract self-contained facts from this text.

RULES:
- Each fact must be complete and understandable on its own (no pronouns, no "yesterday")
- Use third person ("User prefers..." not "I prefer...")
- Include importance 0.0-1.0 (1.0 = critical identity/relationship)
- Skip temporary or trivial information
- Maximum %d facts

CATEGORIES: preference, belief, biographical, event, relationship, task, knowledge, other

OUTPUT FORMAT (one fact per line):
FACT|<category>|<importance>|<fact text>

TEXT:
%s

FACTS:`, maxFacts, preprocessedText)
}

// SimplifiedFactExtractionPrompt is a lighter version for very small models.
func (p *PromptTemplates) SimplifiedFactExtractionPrompt(text string, maxFacts int) string {
	return fmt.Sprintf(`Extract %d important facts about the user.

Each fact should be:
- Self-contained (understandable alone)
- In third person ("User likes...")
- Important for future conversations

OUTPUT FORMAT:
FACT|<type>|<importance 0-1>|<fact>

Types: preference, biographical, relationship, task, other

TEXT: %s

FACTS:`, maxFacts, text)
}

// ============================================================================
// Entity & Relationship Extraction Prompts
// ============================================================================

// EntityExtractionPrompt generates a prompt for extracting entities.
// Note: This is backup for when regex extraction fails; prefer rule-based.
func (p *PromptTemplates) EntityExtractionPrompt(text string) string {
	return fmt.Sprintf(`Extract named entities from this text.

ENTITY TYPES: person, place, organization, concept, thing, time, other

OUTPUT FORMAT (one entity per line):
ENTITY|<type>|<name>

TEXT: %s

ENTITIES:`, text)
}

// RelationshipExtractionPrompt generates a prompt for extracting relationships.
func (p *PromptTemplates) RelationshipExtractionPrompt(fact string, entities []string) string {
	entityList := strings.Join(entities, ", ")
	return fmt.Sprintf(`Extract relationships between entities from this fact.

FACT: %s
ENTITIES: %s

RELATIONSHIP TYPES: knows, works_at, lives_in, owns, likes, dislikes, uses, member_of, related_to

OUTPUT FORMAT (one per line):
REL|<source>|<relation>|<target>|<confidence 0-1>

RELATIONSHIPS:`, fact, entityList)
}

// ============================================================================
// Memory Update Prompts
// ============================================================================

// MemoryUpdatePrompt generates a prompt to decide how to handle a new fact.
func (p *PromptTemplates) MemoryUpdatePrompt(newFact string, existingFacts []string) string {
	existingList := ""
	for i, f := range existingFacts {
		existingList += fmt.Sprintf("[%d] %s\n", i+1, f)
	}

	return fmt.Sprintf(`Decide how to handle this new fact given existing memories.

NEW FACT: %s

EXISTING SIMILAR FACTS:
%s
OPERATIONS:
- ADD: New information not in existing facts
- UPDATE <id>: New fact updates existing fact (provide fact number)
- DELETE <id>: New fact contradicts existing fact (mark old for deletion)
- NOOP: Already known or redundant

OUTPUT FORMAT:
<OPERATION>|<reason>

DECISION:`, newFact, existingList)
}

// FactMergePrompt generates a prompt to merge two related facts.
func (p *PromptTemplates) FactMergePrompt(newFact, existingFact string) string {
	return fmt.Sprintf(`Merge these facts into one complete fact.

NEW: %s
EXISTING: %s

Rules:
- Combine all information
- Keep most recent details if conflicting
- Write as single clear statement in third person

MERGED FACT:`, newFact, existingFact)
}

// ============================================================================
// Summary Prompts
// ============================================================================

// RollingSummaryPrompt generates a prompt for creating a user summary.
func (p *PromptTemplates) RollingSummaryPrompt(facts []string, maxWords int) string {
	factList := ""
	for i, f := range facts {
		factList += fmt.Sprintf("%d. %s\n", i+1, f)
	}

	return fmt.Sprintf(`Create a user profile summary from these facts.

FACTS:
%s
RULES:
- Write in second person ("You are...", "You prefer...")
- Focus on most important characteristics
- Group related information
- Keep under %d words
- Natural, conversational tone

USER SUMMARY:`, factList, maxWords)
}

// IncrementalSummaryPrompt updates an existing summary with new facts.
func (p *PromptTemplates) IncrementalSummaryPrompt(existingSummary string, newFacts []string) string {
	factList := ""
	for i, f := range newFacts {
		factList += fmt.Sprintf("%d. %s\n", i+1, f)
	}

	return fmt.Sprintf(`Update this user summary with new information.

CURRENT SUMMARY:
%s

NEW FACTS:
%s
RULES:
- Integrate new facts naturally
- Update or remove contradicted information
- Keep concise and focused
- Maintain second person perspective

UPDATED SUMMARY:`, existingSummary, factList)
}

// EpisodeSummaryPrompt summarizes a conversation session.
func (p *PromptTemplates) EpisodeSummaryPrompt(turns []ConversationTurn) string {
	var sb strings.Builder
	for _, turn := range turns {
		role := "User"
		if turn.Role == "assistant" {
			role = "Assistant"
		}
		sb.WriteString(fmt.Sprintf("%s: %s\n", role, turn.Content))
	}

	return fmt.Sprintf(`Summarize this conversation session in 2-3 sentences.

CONVERSATION:
%s
Focus on:
- Main topics discussed
- Key decisions or information shared
- Any action items or follow-ups

SESSION SUMMARY:`, sb.String())
}

// ============================================================================
// Helper Functions for Parsing LLM Responses
// ============================================================================

// ExtractedFact represents a fact parsed from LLM output.
type ExtractedFact struct {
	Text       string
	Category   FactCategory
	Importance float64
	Entities   []ExtractedEntity
}

// ExtractedEntity represents an entity parsed from LLM output.
type ExtractedEntity struct {
	Name       string
	EntityType EntityType
}

// ExtractedRelationship represents a relationship parsed from LLM output.
type ExtractedRelationship struct {
	SourceName   string
	RelationType string
	TargetName   string
	Confidence   float64
}

// UpdateOperation defines memory update operations.
type UpdateOperation string

const (
	OpAdd    UpdateOperation = "ADD"
	OpUpdate UpdateOperation = "UPDATE"
	OpDelete UpdateOperation = "DELETE"
	OpNoop   UpdateOperation = "NOOP"
)

// ParseFactExtractionResponse parses the LLM response from fact extraction.
func ParseFactExtractionResponse(response string) []ExtractedFact {
	var facts []ExtractedFact
	lines := strings.Split(response, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "FACT|") {
			continue
		}

		parts := strings.SplitN(line, "|", 4)
		if len(parts) != 4 {
			continue
		}

		category := strings.TrimSpace(parts[1])
		importance := parseFloat(strings.TrimSpace(parts[2]), 0.5)
		text := strings.TrimSpace(parts[3])

		if text == "" {
			continue
		}

		facts = append(facts, ExtractedFact{
			Text:       text,
			Category:   normalizeCategory(category),
			Importance: clampFloat(importance, 0.0, 1.0),
		})
	}

	return facts
}

// ParseEntityExtractionResponse parses the LLM response from entity extraction.
func ParseEntityExtractionResponse(response string) []ExtractedEntity {
	var entities []ExtractedEntity
	lines := strings.Split(response, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "ENTITY|") {
			continue
		}

		parts := strings.SplitN(line, "|", 3)
		if len(parts) != 3 {
			continue
		}

		entType := strings.TrimSpace(parts[1])
		name := strings.TrimSpace(parts[2])

		if name == "" {
			continue
		}

		entities = append(entities, ExtractedEntity{
			Name:       name,
			EntityType: normalizeEntityType(entType),
		})
	}

	return entities
}

// ParseRelationshipExtractionResponse parses the LLM response from relationship extraction.
func ParseRelationshipExtractionResponse(response string) []ExtractedRelationship {
	var relationships []ExtractedRelationship
	lines := strings.Split(response, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "REL|") {
			continue
		}

		parts := strings.SplitN(line, "|", 5)
		if len(parts) != 5 {
			continue
		}

		source := strings.TrimSpace(parts[1])
		relType := strings.TrimSpace(parts[2])
		target := strings.TrimSpace(parts[3])
		confidence := parseFloat(strings.TrimSpace(parts[4]), 0.8)

		if source == "" || target == "" || relType == "" {
			continue
		}

		relationships = append(relationships, ExtractedRelationship{
			SourceName:   source,
			RelationType: relType,
			TargetName:   target,
			Confidence:   clampFloat(confidence, 0.0, 1.0),
		})
	}

	return relationships
}

// ParseMemoryUpdateResponse parses the LLM response from memory update prompt.
func ParseMemoryUpdateResponse(response string) (UpdateOperation, int, string) {
	response = strings.TrimSpace(response)
	parts := strings.SplitN(response, "|", 2)

	opStr := strings.TrimSpace(parts[0])
	reason := ""
	if len(parts) > 1 {
		reason = strings.TrimSpace(parts[1])
	}

	// Parse operation and optional ID
	opParts := strings.Fields(opStr)
	if len(opParts) == 0 {
		return OpNoop, 0, "failed to parse response"
	}

	op := strings.ToUpper(opParts[0])
	targetID := 0

	if len(opParts) > 1 {
		targetID = parseInt(opParts[1], 0)
	}

	switch op {
	case "ADD":
		return OpAdd, 0, reason
	case "UPDATE":
		return OpUpdate, targetID, reason
	case "DELETE":
		return OpDelete, targetID, reason
	case "NOOP":
		return OpNoop, 0, reason
	default:
		return OpNoop, 0, "unknown operation: " + op
	}
}

// ConversationContextPrompt formats conversation turns for extraction.
func (p *PromptTemplates) ConversationContextPrompt(turns []ConversationTurn) string {
	var sb strings.Builder
	for _, turn := range turns {
		role := "User"
		if turn.Role == "assistant" {
			role = "Assistant"
		}
		sb.WriteString(fmt.Sprintf("%s: %s\n", role, turn.Content))
	}
	return sb.String()
}

// ============================================================================
// Helper Functions
// ============================================================================

func parseFloat(s string, defaultVal float64) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultVal
	}

	var f float64
	_, err := fmt.Sscanf(s, "%f", &f)
	if err != nil {
		return defaultVal
	}
	return f
}

func parseInt(s string, defaultVal int) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultVal
	}

	var i int
	_, err := fmt.Sscanf(s, "%d", &i)
	if err != nil {
		return defaultVal
	}
	return i
}

func clampFloat(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func normalizeCategory(s string) FactCategory {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "preference", "preferences", "pref":
		return CategoryPreference
	case "belief", "beliefs", "opinion":
		return CategoryBelief
	case "biographical", "biography", "bio", "identity":
		return CategoryBiographical
	case "event", "events", "experience":
		return CategoryEvent
	case "relationship", "relationships", "relation":
		return CategoryRelationship
	case "task", "tasks", "todo", "goal":
		return CategoryTask
	case "knowledge", "fact", "info":
		return CategoryKnowledge
	default:
		return CategoryOther
	}
}

func normalizeEntityType(s string) EntityType {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "person", "people", "user":
		return EntityPerson
	case "place", "location", "city", "country":
		return EntityPlace
	case "organization", "org", "company", "business":
		return EntityOrganization
	case "concept", "idea", "topic":
		return EntityConcept
	case "thing", "object", "item":
		return EntityThing
	case "time", "date", "datetime", "period":
		return EntityTime
	default:
		return EntityOther
	}
}
