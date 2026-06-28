package pipeline

import "strings"

// extractJSON pulls the first {...} object out of s, stripping markdown code
// fences and any trailing model-specific tokens (e.g. <|改善>).
func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start != -1 && end != -1 && end >= start {
		return s[start : end+1]
	}
	return strings.TrimSpace(s)
}

// cleanTags trims whitespace, drops blank entries, and removes duplicates while
// preserving order — guarding against models that emit junk like [""] or repeats.
func cleanTags(raw []string) []string {
	seen := make(map[string]struct{}, len(raw))
	out := make([]string, 0, len(raw))
	for _, name := range raw {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if _, dup := seen[name]; dup {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	return out
}
