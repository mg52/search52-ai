package ai

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"text/template"
)

type Prompts struct {
	taggingSystem     *template.Template
	taggingUser       *template.Template
	descriptionSystem *template.Template
	descriptionUser   *template.Template
}

func LoadPrompts(dir string) (*Prompts, error) {
	p := &Prompts{}
	var err error

	load := func(name, file string) (*template.Template, error) {
		data, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			return nil, fmt.Errorf("reading prompt %s: %w", file, err)
		}
		t, err := template.New(name).Parse(string(data))
		if err != nil {
			return nil, fmt.Errorf("parsing prompt %s: %w", file, err)
		}
		return t, nil
	}

	if p.taggingSystem, err = load("tagging_system", "tagging_system.template"); err != nil {
		return nil, err
	}
	if p.taggingUser, err = load("tagging_user", "tagging_user.template"); err != nil {
		return nil, err
	}
	if p.descriptionSystem, err = load("description_system", "description_system.template"); err != nil {
		return nil, err
	}
	if p.descriptionUser, err = load("description_user", "description_user.template"); err != nil {
		return nil, err
	}

	return p, nil
}

type TaggingData struct {
	ExistingTags string
	MaxTags      int
	Content      string
}

type DescriptionData struct {
	TagName  string
	Examples string
}

func (p *Prompts) RenderTaggingSystem(data TaggingData) (string, error) {
	return render(p.taggingSystem, data)
}

func (p *Prompts) RenderTaggingUser(data TaggingData) (string, error) {
	return render(p.taggingUser, data)
}

func (p *Prompts) RenderDescriptionSystem(data DescriptionData) (string, error) {
	return render(p.descriptionSystem, data)
}

func (p *Prompts) RenderDescriptionUser(data DescriptionData) (string, error) {
	return render(p.descriptionUser, data)
}

func render(t *template.Template, data any) (string, error) {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("rendering template %s: %w", t.Name(), err)
	}
	return buf.String(), nil
}
