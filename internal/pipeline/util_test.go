package pipeline

import "testing"

func TestExtractJSON(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain", `{"a":1}`, `{"a":1}`},
		{"code fence", "```json\n{\"a\":1}\n```", `{"a":1}`},
		{"trailing garbage", `{"a":1}<|改善>`, `{"a":1}`},
		{"prefix and suffix", `here is json {"a":1} thanks`, `{"a":1}`},
		{"nested", `{"a":{"b":1}}`, `{"a":{"b":1}}`},
		{"no braces", "plain prose answer", "plain prose answer"},
		{"whitespace only", "   spaced   ", "spaced"},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractJSON(c.in); got != c.want {
				t.Errorf("extractJSON(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestCleanTags(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want []string
	}{
		{"empty string dropped", []string{""}, []string{}},
		{"whitespace dropped", []string{"  ", "\t"}, []string{}},
		{"trim and keep", []string{" audio "}, []string{"audio"}},
		{"dedup preserve order", []string{"a", "b", "a"}, []string{"a", "b"}},
		{"mixed junk", []string{"audio", "", " ", "wireless", "audio"}, []string{"audio", "wireless"}},
		{"nil", nil, []string{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := cleanTags(c.in)
			if len(got) != len(c.want) {
				t.Fatalf("cleanTags(%v) = %v, want %v", c.in, got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Errorf("cleanTags(%v)[%d] = %q, want %q", c.in, i, got[i], c.want[i])
				}
			}
		})
	}
}
