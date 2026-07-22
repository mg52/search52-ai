package llm

import (
	"reflect"
	"testing"
)

func TestExtractJSON(t *testing.T) {
	cases := map[string]string{
		`{"a":1}`:                        `{"a":1}`,
		"```json\n{\"a\":1}\n```":        `{"a":1}`,
		"prefix chatter {\"a\":1} trail": `{"a":1}`,
		"no braces here":                 "no braces here",
	}
	for in, want := range cases {
		if got := extractJSON(in); got != want {
			t.Errorf("extractJSON(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCleanNames(t *testing.T) {
	got := cleanNames([]string{" Audio_Devices ", "", "audio_devices", "  ", "Medical"})
	want := []string{"audio_devices", "medical"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("cleanNames = %v, want %v", got, want)
	}
}
