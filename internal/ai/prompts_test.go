package ai

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writePromptDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	return dir
}

func validPromptFiles() map[string]string {
	return map[string]string{
		"tagging_system.template":     "tagging max {{.MaxCategories}}",
		"tagging_user.template":       "existing {{.ExistingCategories}} doc {{.Content}}",
		"description_system.template": "describe {{.CategoryName}}",
		"description_user.template":   "category {{.CategoryName}} ex {{.Examples}}",
	}
}

func TestLoadAndRenderPrompts(t *testing.T) {
	dir := writePromptDir(t, validPromptFiles())
	p, err := LoadPrompts(dir)
	if err != nil {
		t.Fatalf("LoadPrompts: %v", err)
	}

	ts, err := p.RenderTaggingSystem(TaggingData{MaxCategories: 5})
	if err != nil || !strings.Contains(ts, "max 5") {
		t.Errorf("RenderTaggingSystem = %q, err %v", ts, err)
	}
	tu, err := p.RenderTaggingUser(TaggingData{ExistingCategories: "a,b", Content: "hello"})
	if err != nil || !strings.Contains(tu, "a,b") || !strings.Contains(tu, "hello") {
		t.Errorf("RenderTaggingUser = %q, err %v", tu, err)
	}
	ds, err := p.RenderDescriptionSystem(DescriptionData{CategoryName: "audio"})
	if err != nil || !strings.Contains(ds, "audio") {
		t.Errorf("RenderDescriptionSystem = %q, err %v", ds, err)
	}
	du, err := p.RenderDescriptionUser(DescriptionData{CategoryName: "audio", Examples: "ex1"})
	if err != nil || !strings.Contains(du, "ex1") {
		t.Errorf("RenderDescriptionUser = %q, err %v", du, err)
	}
}

func TestLoadPromptsMissingFile(t *testing.T) {
	files := validPromptFiles()
	delete(files, "tagging_user.template")
	dir := writePromptDir(t, files)
	if _, err := LoadPrompts(dir); err == nil {
		t.Fatal("expected error for missing prompt file")
	}
}

func TestLoadPromptsBadTemplate(t *testing.T) {
	files := validPromptFiles()
	files["tagging_system.template"] = "broken {{.MaxCategories" // unterminated action
	dir := writePromptDir(t, files)
	if _, err := LoadPrompts(dir); err == nil {
		t.Fatal("expected parse error for bad template")
	}
}

func TestRenderUnknownFieldErrors(t *testing.T) {
	files := validPromptFiles()
	files["tagging_system.template"] = "{{.DoesNotExist}}"
	dir := writePromptDir(t, files)
	p, err := LoadPrompts(dir)
	if err != nil {
		t.Fatalf("LoadPrompts: %v", err)
	}
	if _, err := p.RenderTaggingSystem(TaggingData{}); err == nil {
		t.Fatal("expected render error for unknown field")
	}
}
