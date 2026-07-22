package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"
	"time"

	"github.com/mg52/search52-ai/internal/engine"
)

// indexNamePattern restricts index names to a safe, path-friendly character set
// so they can be used directly as directory names.
var indexNamePattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)

// Manager owns every index and the shared clients (embedder, LLM). Each
// index is a self-contained engine.Index — embedding-clustered or
// LLM-categorized — persisted under dataDir/<name>/.
type Manager struct {
	mu       sync.RWMutex
	dataDir  string
	deps     engine.Deps
	defaults engine.Config
	indexes  map[string]engine.Index
}

func NewManager(dataDir string, deps engine.Deps, defaults engine.Config) *Manager {
	return &Manager{
		dataDir:  dataDir,
		deps:     deps,
		defaults: defaults,
		indexes:  make(map[string]engine.Index),
	}
}

// LoadExisting scans dataDir for index directories and loads each snapshot into
// memory. Missing dataDir is not an error (fresh start).
func (m *Manager) LoadExisting() error {
	entries, err := os.ReadDir(m.dataDir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read data dir %s: %w", m.dataDir, err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(m.dataDir, e.Name())
		idx, err := engine.Load(dir, m.deps)
		if err != nil {
			log.Printf("manager: skipping %s: %v", e.Name(), err)
			continue
		}
		m.indexes[idx.Name()] = idx
		log.Printf("manager: loaded %s index %q (%d docs, %d categories)", idx.Kind(), idx.Name(), idx.DocCount(), idx.CategoryCount())
	}
	return nil
}

func (m *Manager) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /indexes", m.createIndex)
	mux.HandleFunc("GET /indexes", m.listIndexes)
	mux.HandleFunc("DELETE /indexes/{index}", m.deleteIndex)

	mux.HandleFunc("POST /indexes/{index}/persist", m.persistIndex)

	mux.HandleFunc("POST /indexes/{index}/documents", m.createDocument)
	mux.HandleFunc("POST /indexes/{index}/documents/batch", m.batchDocuments)
	mux.HandleFunc("GET /indexes/{index}/documents/{id}", m.getDocument)
	mux.HandleFunc("PUT /indexes/{index}/documents/{id}", m.updateDocument)
	mux.HandleFunc("DELETE /indexes/{index}/documents/{id}", m.deleteDocument)

	mux.HandleFunc("GET /indexes/{index}/categories", m.listCategories)
	mux.HandleFunc("GET /indexes/{index}/categories/{name}", m.getCategory)

	mux.HandleFunc("POST /indexes/{index}/search", m.search)

	mux.HandleFunc("GET /health", m.health)
	return mux
}

// index resolves the {index} path value to a live engine, writing a 404/400 and
// returning nil when it cannot.
func (m *Manager) index(w http.ResponseWriter, r *http.Request) engine.Index {
	name := r.PathValue("index")
	if !indexNamePattern.MatchString(name) {
		jsonError(w, "invalid index name", http.StatusBadRequest)
		return nil
	}
	m.mu.RLock()
	idx := m.indexes[name]
	m.mu.RUnlock()
	if idx == nil {
		jsonError(w, "index not found", http.StatusNotFound)
		return nil
	}
	return idx
}

// -------------------- indexes --------------------

type createIndexReq struct {
	Name                string  `json:"name"`
	Kind                string  `json:"kind"` // "embedding" (default) or "llm"
	CategoryThreshold   float64 `json:"category_threshold"`
	MaxCategoriesPerDoc int     `json:"max_categories_per_doc"`
	MaxCategories       int     `json:"max_categories"`
	TopNCategories      int     `json:"top_n_categories"`
	VarianceThreshold   float64 `json:"variance_threshold"`
	VarianceMinCount    int     `json:"variance_min_count"`
	DisableSplit        *bool   `json:"disable_split"`
	MaxDocsPerCategory  int     `json:"max_docs_per_category"`
}

func (m *Manager) createIndex(w http.ResponseWriter, r *http.Request) {
	var req createIndexReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if !indexNamePattern.MatchString(req.Name) {
		jsonError(w, "name must be 1-128 chars of letters, numbers, _.- ", http.StatusBadRequest)
		return
	}

	// Fall back to server defaults for any tuning field left at zero.
	cfg := m.defaults
	if req.CategoryThreshold > 0 {
		cfg.CategoryThreshold = req.CategoryThreshold
	}
	if req.MaxCategoriesPerDoc > 0 {
		cfg.MaxCategoriesPerDoc = req.MaxCategoriesPerDoc
	}
	if req.MaxCategories > 0 {
		cfg.MaxCategories = req.MaxCategories
	}
	if req.TopNCategories > 0 {
		cfg.TopNCategories = req.TopNCategories
	}
	if req.VarianceThreshold > 0 {
		cfg.VarianceThreshold = req.VarianceThreshold
	}
	if req.VarianceMinCount > 0 {
		cfg.VarianceMinCount = req.VarianceMinCount
	}
	if req.DisableSplit != nil {
		cfg.DisableSplit = *req.DisableSplit
	}
	if req.MaxDocsPerCategory > 0 {
		cfg.MaxDocsPerCategory = req.MaxDocsPerCategory
	}

	m.mu.Lock()
	if _, exists := m.indexes[req.Name]; exists {
		m.mu.Unlock()
		jsonError(w, "index already exists", http.StatusConflict)
		return
	}
	idx, err := engine.New(engine.Kind(req.Kind), req.Name, m.deps, cfg)
	if err != nil {
		m.mu.Unlock()
		jsonError(w, err.Error(), http.StatusBadRequest)
		return
	}
	m.indexes[req.Name] = idx
	m.mu.Unlock()

	jsonResponse(w, map[string]any{"name": idx.Name(), "kind": idx.Kind()}, http.StatusCreated)
}

func (m *Manager) listIndexes(w http.ResponseWriter, r *http.Request) {
	m.mu.RLock()
	indexes := make([]engine.Index, 0, len(m.indexes))
	for _, idx := range m.indexes {
		indexes = append(indexes, idx)
	}
	m.mu.RUnlock()

	type indexSummary struct {
		Name          string `json:"name"`
		Kind          string `json:"kind"`
		DocCount      int    `json:"doc_count"`
		CategoryCount int    `json:"category_count"`
	}
	summaries := make([]indexSummary, len(indexes))
	for i, idx := range indexes {
		summaries[i] = indexSummary{Name: idx.Name(), Kind: idx.Kind(), DocCount: idx.DocCount(), CategoryCount: idx.CategoryCount()}
	}
	sort.Slice(summaries, func(i, j int) bool { return summaries[i].Name < summaries[j].Name })
	jsonResponse(w, map[string]any{"indexes": summaries, "total": len(summaries)}, http.StatusOK)
}

func (m *Manager) deleteIndex(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("index")
	if !indexNamePattern.MatchString(name) {
		jsonError(w, "invalid index name", http.StatusBadRequest)
		return
	}
	m.mu.Lock()
	_, ok := m.indexes[name]
	delete(m.indexes, name)
	m.mu.Unlock()
	if !ok {
		jsonError(w, "index not found", http.StatusNotFound)
		return
	}
	if err := os.RemoveAll(filepath.Join(m.dataDir, name)); err != nil {
		log.Printf("manager: remove index dir %q: %v", name, err)
	}
	w.WriteHeader(http.StatusNoContent)
}

// persistIndex writes the named index's current in-memory state to disk on
// demand. Document ingestion no longer persists after every write; callers
// must invoke this endpoint to snapshot the index.
func (m *Manager) persistIndex(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	if err := idx.Save(filepath.Join(m.dataDir, idx.Name())); err != nil {
		log.Printf("manager: persist %q: %v", idx.Name(), err)
		jsonError(w, "persist failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, map[string]any{"name": idx.Name(), "persisted": true}, http.StatusOK)
}

// -------------------- documents --------------------

type documentReq struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

func (m *Manager) createDocument(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	var req documentReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Content == "" {
		jsonError(w, "content is required", http.StatusBadRequest)
		return
	}
	if req.ID == "" {
		req.ID = fmt.Sprintf("%d", time.Now().UnixNano())
	}
	if _, exists := idx.GetDocument(req.ID); exists {
		jsonError(w, "document already exists", http.StatusConflict)
		return
	}
	doc, err := idx.Process(r.Context(), req.ID, req.Content)
	if err != nil {
		log.Printf("handler: categorize %s: %v", req.ID, err)
		jsonError(w, "categorization failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, doc, http.StatusCreated)
}

type batchDocumentsReq struct {
	Documents []documentReq `json:"documents"`
}

type batchError struct {
	ID    string `json:"id"`
	Error string `json:"error"`
}

// batchDocuments ingests many documents in one request. Every document is
// categorized in order. Ingestion no longer persists to disk; call POST
// /indexes/{index}/persist once the batch is done. Individual failures
// (empty content, categorization errors) are collected and reported without
// aborting the rest of the batch. A document whose id already exists is
// re-processed (its content and category memberships are replaced), so a
// batch acts as an upsert.
func (m *Manager) batchDocuments(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	var req batchDocumentsReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if len(req.Documents) == 0 {
		jsonError(w, "documents is required", http.StatusBadRequest)
		return
	}

	indexed := 0
	errs := []batchError{}
	for i, d := range req.Documents {
		id := d.ID
		if id == "" {
			id = fmt.Sprintf("%d-%d", time.Now().UnixNano(), i)
		}
		if d.Content == "" {
			errs = append(errs, batchError{ID: id, Error: "content is required"})
			continue
		}
		if _, err := idx.Process(r.Context(), id, d.Content); err != nil {
			errs = append(errs, batchError{ID: id, Error: err.Error()})
			continue
		}
		indexed++
	}

	status := http.StatusCreated
	if indexed == 0 {
		status = http.StatusBadRequest
	}
	jsonResponse(w, map[string]any{"indexed": indexed, "failed": len(errs), "errors": errs}, status)
}

func (m *Manager) getDocument(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	doc, ok := idx.GetDocument(r.PathValue("id"))
	if !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	jsonResponse(w, doc, http.StatusOK)
}

func (m *Manager) updateDocument(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	id := r.PathValue("id")
	if _, ok := idx.GetDocument(id); !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	var req documentReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Content == "" {
		jsonError(w, "content is required", http.StatusBadRequest)
		return
	}
	doc, err := idx.Process(r.Context(), id, req.Content)
	if err != nil {
		log.Printf("handler: updating doc %s: %v", id, err)
		jsonError(w, "categorization failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, doc, http.StatusOK)
}

func (m *Manager) deleteDocument(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	if _, ok := idx.Remove(r.PathValue("id")); !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// -------------------- categories --------------------

func (m *Manager) listCategories(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	cats := idx.ListCategories()
	sort.Slice(cats, func(i, j int) bool { return cats[i].Name < cats[j].Name })
	jsonResponse(w, cats, http.StatusOK)
}

func (m *Manager) getCategory(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	c, ok := idx.GetCategory(r.PathValue("name"))
	if !ok {
		jsonError(w, "category not found", http.StatusNotFound)
		return
	}
	jsonResponse(w, c, http.StatusOK)
}

// -------------------- search --------------------

type searchReq struct {
	Q     string `json:"q"`
	Limit int    `json:"limit"`
}

func (m *Manager) search(w http.ResponseWriter, r *http.Request) {
	idx := m.index(w, r)
	if idx == nil {
		return
	}
	var req searchReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Q == "" {
		jsonError(w, "q is required", http.StatusBadRequest)
		return
	}
	results, err := idx.Search(r.Context(), req.Q, req.Limit)
	if err != nil {
		log.Printf("handler: search error: %v", err)
		jsonError(w, "search failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, results, http.StatusOK)
}

// -------------------- misc --------------------

func (m *Manager) health(w http.ResponseWriter, r *http.Request) {
	m.mu.RLock()
	count := len(m.indexes)
	m.mu.RUnlock()
	jsonResponse(w, map[string]any{"status": "ok", "index_count": count}, http.StatusOK)
}

func jsonResponse(w http.ResponseWriter, data any, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, msg string, status int) {
	jsonResponse(w, map[string]string{"error": msg}, status)
}
