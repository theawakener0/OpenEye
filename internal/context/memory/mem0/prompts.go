package mem0

import (
	"fmt"
	"strings"
)

// PromptTemplates contains all prompts used by the mem0 system.
// These prompts are optimized for small language models (SLMs) with:
// - Clear, structured output formats
// - Explicit field names and delimiters
// - Short, focused instructions
// - Examples when beneficial
type PromptTemplates struct{}

// NewPromptTemplates creates a new prompt template provider.
func NewPromptTemplates() *PromptTemplates {
	return &PromptTemplates{}
}

// FactExtractionPrompt generates a prompt for extracting facts from conversation.
// Optimized for SLMs: uses simple line-based format instead of JSON.
func (p *PromptTemplates) FactExtractionPrompt(conversation string, maxFacts int) string {
	return fmt.Sprintf(`Extract important facts about the user from this conversation.

RULES:
- Extract ONLY facts about the user (preferences, beliefs, biographical info, relationships, tasks)
- Each fact must be a single, atomic statement
- Use third person ("User prefers..." not "I prefer...")
- Include importance 0.0-1.0 (1.0 = critical personal info)
- Skip trivial/temporary information
- Maximum %d facts

CATEGORIES: preference, belief, biographical, event, relationship, task, other

OUTPUT FORMAT (one fact per line):
FACT|<category>|<importance>|<fact text>

CONVERSATION:
%s

FACTS:`, maxFacts, conversation)
}

// EntityExtractionPrompt generates a prompt for extracting entities from text.
func (p *PromptTemplates) EntityExtractionPrompt(text string) string {
	return fmt.Sprintf(`Extract named entities from this text.

ENTITY TYPES: person, place, thing, concept, organization, time, other

OUTPUT FORMAT (one entity per line):
ENTITY|<type>|<name>

TEXT:
%s

ENTITIES:`, text)
}

// RelationshipExtractionPrompt generates a prompt for extracting relationships.
func (p *PromptTemplates) RelationshipExtractionPrompt(fact string, entities []string) string {
	entityList := strings.Join(entities, ", ")
	return fmt.Sprintf(`Given this fact and entities, extract relationships between entities.

FACT: %s
KNOWN ENTITIES: %s

RELATIONSHIP TYPES: knows, works_at, lives_in, owns, likes, dislikes, uses, created, is_member_of, related_to

OUTPUT FORMAT (one relationship per line):
REL|<source entity>|<relation type>|<target entity>|<confidence 0.0-1.0>

RELATIONSHIPS:`, fact, entityList)
}

// MemoryUpdatePrompt generates a prompt to decide ADD/UPDATE/DELETE/NOOP.
func (p *PromptTemplates) MemoryUpdatePrompt(newFact string, existingFacts []string) string {
	existingList := ""
	for i, f := range existingFacts {
		existingList += fmt.Sprintf("[%d] %s\n", i+1, f)
	}

	return fmt.Sprintf(`Decide how to handle a new fact given existing memories.

NEW FACT: %s

EXISTING SIMILAR FACTS:
%s
OPERATIONS:
- ADD: New fact provides new information
- UPDATE <id>: New fact updates/refines an existing fact (provide the fact number to update)
- DELETE <id>: New fact contradicts/invalidates an existing fact (mark for deletion)
- NOOP: New fact is already known or redundant

Respond with exactly one operation and brief reason.

OUTPUT FORMAT:
<OPERATION>|<reason>

Examples:
ADD|This is new information about user's job
UPDATE 2|Refines the location information in fact 2
DELETE 1|Contradicts fact 1, user changed preference
NOOP|Already captured in existing facts

DECISION:`, newFact, existingList)
}

// MemoryUpdateWithMergePrompt handles updates that require merging facts.
func (p *PromptTemplates) MemoryUpdateWithMergePrompt(newFact string, existingFact string) string {
	return fmt.Sprintf(`Merge these two facts into a single, more complete fact.

NEW FACT: %s
EXISTING FACT: %s

Rules:
- Combine information from both facts
- Keep the most recent/specific information if conflicting
- Write as a single, clear statement
- Use third person ("User...")

MERGED FACT:`, newFact, existingFact)
}

// RollingSummaryPrompt generates a prompt for creating/updating the rolling summary.
func (p *PromptTemplates) RollingSummaryPrompt(facts []string, maxTokens int) string {
	factList := ""
	for i, f := range facts {
		factList += fmt.Sprintf("%d. %s\n", i+1, f)
	}

	return fmt.Sprintf(`Create a concise summary of the user based on these facts.

FACTS:
%s
RULES:
- Write in second person ("You are...", "You prefer...")
- Focus on most important characteristics
- Group related information
- Keep under %d words
- Natural, conversational tone

USER SUMMARY:`, factList, maxTokens/4) // Rough word estimate
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
- Remove contradicted information
- Keep concise and focused
- Maintain second person perspective

UPDATED SUMMARY:`, existingSummary, factList)
}

// ImportanceEvaluationPrompt helps evaluate the importance of a fact.
func (p *PromptTemplates) ImportanceEvaluationPrompt(fact string) string {
	return fmt.Sprintf(`Rate the long-term importance of this fact about the user on a scale of 0.0 to 1.0.

FACT: %s

SCORING GUIDE:
- 0.9-1.0: Core identity (name, critical relationships, major life events)
- 0.7-0.8: Important preferences or recurring patterns
- 0.5-0.6: Useful context or moderate preferences  
- 0.3-0.4: Minor details that might be relevant
- 0.1-0.2: Temporary or trivial information

Respond with just the number.

IMPORTANCE:`, fact)
}

// CategoryClassificationPrompt classifies a fact into a category.
func (p *PromptTemplates) CategoryClassificationPrompt(fact string) string {
	return fmt.Sprintf(`Classify this fact into exactly one category.

FACT: %s

CATEGORIES:
- preference: Likes, dislikes, choices, favorites
- belief: Opinions, values, worldviews
- biographical: Personal details, history, identity
- event: Things that happened, experiences
- relationship: Connections to people, groups
- task: Goals, todos, intentions
- other: Doesn't fit other categories

Respond with just the category name.

CATEGORY:`, fact)
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

// ConversationTurn represents a single turn in a conversation.
type ConversationTurn struct {
	Role    string // "user" or "assistant"
	Content string
	TurnID  string // Optional identifier for the turn
}

// ParseFactExtractionResponse parses the LLM response from FactExtractionPrompt.
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

// ExtractedFact represents a fact parsed from LLM output.
type ExtractedFact struct {
	Text       string
	Category   FactCategory
	Importance float64
	Entities   []ExtractedEntity
}

// ParseEntityExtractionResponse parses the LLM response from EntityExtractionPrompt.
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

// ExtractedEntity represents an entity parsed from LLM output.
type ExtractedEntity struct {
	Name       string
	EntityType EntityType
}

// ParseRelationshipExtractionResponse parses the LLM response from RelationshipExtractionPrompt.
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

// ExtractedRelationship represents a relationship parsed from LLM output.
type ExtractedRelationship struct {
	SourceName   string
	RelationType string
	TargetName   string
	Confidence   float64
}

// ParseMemoryUpdateResponse parses the LLM response from MemoryUpdatePrompt.
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

// Helper functions

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
	case "preference", "preferences":
		return CategoryPreference
	case "belief", "beliefs":
		return CategoryBelief
	case "biographical", "biography", "bio":
		return CategoryBiographical
	case "event", "events":
		return CategoryEvent
	case "relationship", "relationships":
		return CategoryRelationship
	case "task", "tasks", "todo":
		return CategoryTask
	default:
		return CategoryOther
	}
}

func normalizeEntityType(s string) EntityType {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "person", "people":
		return EntityPerson
	case "place", "location":
		return EntityPlace
	case "thing", "object":
		return EntityThing
	case "concept", "idea":
		return EntityConcept
	case "organization", "org", "company":
		return EntityOrganization
	case "time", "date", "datetime":
		return EntityTime
	default:
		return EntityOther
	}
}
