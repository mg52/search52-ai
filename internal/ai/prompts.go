package ai

import (
	"fmt"
	"strconv"
	"strings"
)

// prompt is used for both documents and search queries, so both end up in the
// same label vocabulary. The text to label is the user message.
const prompt = `You are the indexing part of a search engine. You will get a short text. It is either a document or a search query.
Reply with labels (search keywords) for the text.

Write the labels in this order:
1. First the broad categories the thing belongs to (for example "kitchen", "audio", "garden"), so that a general search also finds it.
2. Then the specific and detailed labels: the main thing, its type, its features, and the brand name if there is one.

Rules:
- Return from 1 to {{max}} labels. You do not have to reach {{max}}: fewer labels are fine when the text needs fewer. Every label must be relevant to the text.
- {{words}}
- Use simple, plain words that anyone understands. Make the labels as simple and clear as possible.
- Always write the labels in English, even if the text is in another language.
- Output ONLY the labels on a single line, separated by commas. No explanation, no numbering, no JSON.

Example text: Fellow electric kettle with variable temperature control
Example output: kitchen, appliance, kettle, electric, fellow`

// existingLabelsBlock is appended to prompt by PromptWithLabels. prompt itself
// is left exactly as it is.
const existingLabelsBlock = `

Labels already used in the index, most used first:
{{labels}}

When one of these labels fits the text, use it exactly as written instead of inventing a similar one. Only add a new label when none of them fits.`

// Prompt is the system prompt that turns a document or a search query into
// 1 to maxLabels labels of at most maxWords words each (<= 0: any length), broad
// categories first.
func Prompt(maxLabels, maxWords int) string {
	return strings.NewReplacer("{{max}}", strconv.Itoa(maxLabels), "{{words}}", wordsRule(maxWords)).Replace(prompt)
}

func wordsRule(maxWords int) string {
	switch {
	case maxWords == 1:
		return "Each label is exactly one word, lowercase, with no spaces or hyphens. Never use two words in one label."
	case maxWords > 1:
		return fmt.Sprintf("Each label is 1 to %d words, lowercase. Never more than %d words.", maxWords, maxWords)
	}
	return "Each label is a short word or phrase, lowercase."
}

// PromptWithLabels is Prompt followed by the labels the index already has, which
// the caller passes most used first. It is for when the LLM should reuse those
// labels instead of inventing synonyms. Prompt is unchanged and is a prefix of
// this text.
func PromptWithLabels(maxLabels, maxWords int, existing []string) string {
	list := "(none yet)"
	if len(existing) > 0 {
		list = strings.Join(existing, ", ")
	}
	return Prompt(maxLabels, maxWords) + strings.ReplaceAll(existingLabelsBlock, "{{labels}}", list)
}
