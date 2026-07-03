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

// Manager owns every index and the shared embedder. Each index is a
// self-contained engine.SearchEngine persisted under dataDir/<name>/.
type Manager struct {
	mu       sync.RWMutex
	dataDir  string
	embedder engine.Embedder
	defaults engine.Config
	indexes  map[string]*engine.SearchEngine
}

func NewManager(dataDir string, embedder engine.Embedder, defaults engine.Config) *Manager {
	return &Manager{
		dataDir:  dataDir,
		embedder: embedder,
		defaults: defaults,
		indexes:  make(map[string]*engine.SearchEngine),
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
		se, err := engine.Load(dir, m.embedder)
		if err != nil {
			log.Printf("manager: skipping %s: %v", e.Name(), err)
			continue
		}
		m.indexes[se.Name()] = se
		log.Printf("manager: loaded index %q (%d docs, %d categories)", se.Name(), se.DocCount(), se.CategoryCount())
	}
	return nil
}

func (m *Manager) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /indexes", m.createIndex)
	mux.HandleFunc("GET /indexes", m.listIndexes)
	mux.HandleFunc("DELETE /indexes/{index}", m.deleteIndex)

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
func (m *Manager) index(w http.ResponseWriter, r *http.Request) *engine.SearchEngine {
	name := r.PathValue("index")
	if !indexNamePattern.MatchString(name) {
		jsonError(w, "invalid index name", http.StatusBadRequest)
		return nil
	}
	m.mu.RLock()
	se := m.indexes[name]
	m.mu.RUnlock()
	if se == nil {
		jsonError(w, "index not found", http.StatusNotFound)
		return nil
	}
	return se
}

// persist writes an index snapshot to disk, logging (but not failing the
// request on) errors — the in-memory state is already updated.
func (m *Manager) persist(se *engine.SearchEngine) {
	if err := se.Save(filepath.Join(m.dataDir, se.Name())); err != nil {
		log.Printf("manager: persist %q: %v", se.Name(), err)
	}
}

// -------------------- indexes --------------------

type createIndexReq struct {
	Name                string  `json:"name"`
	CategoryThreshold   float64 `json:"category_threshold"`
	MaxCategoriesPerDoc int     `json:"max_categories_per_doc"`
	MaxCategories       int     `json:"max_categories"`
	TopNCategories      int     `json:"top_n_categories"`
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

	m.mu.Lock()
	if _, exists := m.indexes[req.Name]; exists {
		m.mu.Unlock()
		jsonError(w, "index already exists", http.StatusConflict)
		return
	}
	se := engine.New(req.Name, m.embedder, cfg)
	m.indexes[req.Name] = se
	m.mu.Unlock()

	m.persist(se)
	jsonResponse(w, map[string]any{"name": se.Name()}, http.StatusCreated)
}

func (m *Manager) listIndexes(w http.ResponseWriter, r *http.Request) {
	m.mu.RLock()
	names := make([]string, 0, len(m.indexes))
	engines := make([]*engine.SearchEngine, 0, len(m.indexes))
	for name, se := range m.indexes {
		names = append(names, name)
		engines = append(engines, se)
	}
	m.mu.RUnlock()

	type indexSummary struct {
		Name          string `json:"name"`
		DocCount      int    `json:"doc_count"`
		CategoryCount int    `json:"category_count"`
	}
	summaries := make([]indexSummary, len(engines))
	for i, se := range engines {
		summaries[i] = indexSummary{Name: se.Name(), DocCount: se.DocCount(), CategoryCount: se.CategoryCount()}
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

// -------------------- documents --------------------

type documentReq struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

func (m *Manager) createDocument(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
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
	if _, exists := se.GetDocument(req.ID); exists {
		jsonError(w, "document already exists", http.StatusConflict)
		return
	}
	doc, err := se.Process(r.Context(), req.ID, req.Content)
	if err != nil {
		log.Printf("handler: categorize %s: %v", req.ID, err)
		jsonError(w, "categorization failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	m.persist(se)
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
// embedded and clustered in order, then the index is persisted exactly ONCE at
// the end — not per document — so a bulk load pays a single snapshot write
// instead of one per item. Individual failures (empty content, embedding
// errors) are collected and reported without aborting the rest of the batch.
// A document whose id already exists is re-processed (its content and category
// memberships are replaced), so a batch acts as an upsert.
func (m *Manager) batchDocuments(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
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
		if _, err := se.Process(r.Context(), id, d.Content); err != nil {
			errs = append(errs, batchError{ID: id, Error: err.Error()})
			continue
		}
		indexed++
	}

	// Persist the whole batch in a single snapshot write — the point of this
	// endpoint. Skip it if nothing was indexed so a fully-failed batch leaves
	// disk untouched.
	if indexed > 0 {
		m.persist(se)
	}

	status := http.StatusCreated
	if indexed == 0 {
		status = http.StatusBadRequest
	}
	jsonResponse(w, map[string]any{"indexed": indexed, "failed": len(errs), "errors": errs}, status)
}

func (m *Manager) getDocument(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
		return
	}
	doc, ok := se.GetDocument(r.PathValue("id"))
	if !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	jsonResponse(w, doc, http.StatusOK)
}

func (m *Manager) updateDocument(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
		return
	}
	id := r.PathValue("id")
	if _, ok := se.GetDocument(id); !ok {
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
	doc, err := se.Process(r.Context(), id, req.Content)
	if err != nil {
		log.Printf("handler: updating doc %s: %v", id, err)
		jsonError(w, "categorization failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	m.persist(se)
	jsonResponse(w, doc, http.StatusOK)
}

func (m *Manager) deleteDocument(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
		return
	}
	if _, ok := se.Remove(r.PathValue("id")); !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	m.persist(se)
	w.WriteHeader(http.StatusNoContent)
}

// -------------------- categories --------------------

func (m *Manager) listCategories(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
		return
	}
	cats := se.ListCategories()
	sort.Slice(cats, func(i, j int) bool { return cats[i].Name < cats[j].Name })
	type categorySummary struct {
		Name      string    `json:"name"`
		DocCount  int       `json:"doc_count"`
		CreatedAt time.Time `json:"created_at"`
	}
	summaries := make([]categorySummary, len(cats))
	for i, c := range cats {
		summaries[i] = categorySummary{Name: c.Name, DocCount: se.DocCountByCategory(c.Name), CreatedAt: c.CreatedAt}
	}
	jsonResponse(w, summaries, http.StatusOK)
}

func (m *Manager) getCategory(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
		return
	}
	name := r.PathValue("name")
	c, ok := se.GetCategory(name)
	if !ok {
		jsonError(w, "category not found", http.StatusNotFound)
		return
	}
	jsonResponse(w, map[string]any{
		"name":        c.Name,
		"doc_count":   se.DocCountByCategory(name),
		"vector_dims": len(c.Centroid),
		"created_at":  c.CreatedAt,
	}, http.StatusOK)
}

// -------------------- search --------------------

type searchReq struct {
	Q     string `json:"q"`
	Limit int    `json:"limit"`
}

func (m *Manager) search(w http.ResponseWriter, r *http.Request) {
	se := m.index(w, r)
	if se == nil {
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
	results, err := se.Search(r.Context(), req.Q, req.Limit)
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
