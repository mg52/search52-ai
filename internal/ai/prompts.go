package ai

import (
	"strconv"
	"strings"
)

const labelsFormat = `Output ONLY the labels, lowercase, separated by commas, on a single line. No JSON, no markdown, no numbering, no explanation.
Example: label1, label2, label3`

const indexPrompt = `You are a search indexing system. You will be given a document: a description and, if present, its category and other details.
Produce the labels a user might search for to find it.

Existing labels already used in the index:
{{labels}}

Rules:
- Return between {{min}} and {{max}} labels.
- ALWAYS reuse an existing label when it fits the document (e.g. if "earphones" exists, do not invent "earbuds"). Create a new label only when no existing one covers the topic.
- Mix broad labels (e.g. "audio", "sound", "earphones") with specific ones (e.g. "noise cancelling earphones").
- Labels are topics or product types, not brand names or one-off attributes.

` + labelsFormat

const searchPrompt = `You are a search system. You will be given a search query.
Produce the labels of the documents the user is looking for.

Existing labels in the index:
{{labels}}

Rules:
- Return between {{min}} and {{max}} labels.
- Pick the existing labels that match the query. Only if none of them fit, return the closest new label.
- Labels are topics or product types, not brand names or one-off attributes.

` + labelsFormat

// IndexPrompt is the system prompt that labels a document, steering the LLM
// toward the labels the index already has and asking for min to max labels.
// The document is the user message.
func IndexPrompt(existing []string, min, max int) string {
	return render(indexPrompt, existing, min, max)
}

// SearchPrompt is the system prompt that labels a search query against the
// labels the index already has, asking for min to max labels. The query is the
// user message.
func SearchPrompt(existing []string, min, max int) string {
	return render(searchPrompt, existing, min, max)
}

func render(tmpl string, existing []string, min, max int) string {
	list := "(none yet)"
	if len(existing) > 0 {
		list = strings.Join(existing, ", ")
	}
	return strings.NewReplacer(
		"{{labels}}", list,
		"{{min}}", strconv.Itoa(min),
		"{{max}}", strconv.Itoa(max),
	).Replace(tmpl)
}
