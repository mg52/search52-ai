package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/mg52/search52-ai/internal/pipeline"
	"github.com/mg52/search52-ai/internal/search"
	"github.com/mg52/search52-ai/internal/store"
)

type Handler struct {
	store       store.Store
	categorizer *pipeline.Categorizer
	searcher    *search.Engine
}

func New(st store.Store, categorizer *pipeline.Categorizer, searcher *search.Engine) *Handler {
	return &Handler{store: st, categorizer: categorizer, searcher: searcher}
}

func (h *Handler) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /documents", h.createDocument)
	mux.HandleFunc("GET /documents/{id}", h.getDocument)
	mux.HandleFunc("PUT /documents/{id}", h.updateDocument)
	mux.HandleFunc("DELETE /documents/{id}", h.deleteDocument)
	mux.HandleFunc("GET /documents", h.listDocuments)
	mux.HandleFunc("GET /categories/{name}", h.getCategory)
	mux.HandleFunc("GET /categories", h.listCategories)
	mux.HandleFunc("POST /search", h.search)
	mux.HandleFunc("GET /health", h.health)
	return mux
}

type documentReq struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

// createDocument embeds the document and clusters it into categories. No LLM is
// involved; categories are discovered incrementally by vector similarity.
func (h *Handler) createDocument(w http.ResponseWriter, r *http.Request) {
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
	if _, exists := h.store.GetDocument(req.ID); exists {
		jsonError(w, "document already exists", http.StatusConflict)
		return
	}

	doc := store.Document{ID: req.ID, Content: req.Content, CreatedAt: time.Now()}
	doc, err := h.categorizer.Process(r.Context(), doc)
	if err != nil {
		log.Printf("handler: categorize %s: %v", doc.ID, err)
		jsonError(w, "categorization failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, doc, http.StatusCreated)
}

func (h *Handler) getDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	doc, ok := h.store.GetDocument(id)
	if !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	jsonResponse(w, doc, http.StatusOK)
}

// updateDocument replaces a document's content and re-clusters it so its
// categories (and the inverted index) stay consistent with the new content.
func (h *Handler) updateDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	doc, ok := h.store.GetDocument(id)
	if !ok {
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

	doc.Content = req.Content
	doc, err := h.categorizer.Process(r.Context(), doc)
	if err != nil {
		log.Printf("handler: updating doc %s: %v", id, err)
		jsonError(w, "categorization failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, doc, http.StatusOK)
}

// deleteDocument removes a document and detaches it from its categories
// (pruning any left empty) via the categorizer, which owns the centroid math.
func (h *Handler) deleteDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := h.categorizer.Remove(id); !ok {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) listDocuments(w http.ResponseWriter, r *http.Request) {
	page := queryInt(r, "page", 1)
	size := queryInt(r, "size", 20)
	if page < 1 {
		page = 1
	}
	if size < 1 || size > 100 {
		size = 20
	}
	docs, total := h.store.ListDocuments(page, size)
	if docs == nil {
		docs = []store.Document{}
	}
	jsonResponse(w, map[string]any{
		"documents": docs,
		"total":     total,
		"page":      page,
		"size":      size,
	}, http.StatusOK)
}

func (h *Handler) listCategories(w http.ResponseWriter, r *http.Request) {
	cats := h.store.ListCategories()
	type categorySummary struct {
		Name      string    `json:"name"`
		DocCount  int       `json:"doc_count"`
		CreatedAt time.Time `json:"created_at"`
	}
	summaries := make([]categorySummary, len(cats))
	for i, c := range cats {
		summaries[i] = categorySummary{
			Name:      c.Name,
			DocCount:  h.store.DocCountByCategory(c.Name),
			CreatedAt: c.CreatedAt,
		}
	}
	jsonResponse(w, summaries, http.StatusOK)
}

func (h *Handler) getCategory(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	c, ok := h.store.GetCategory(name)
	if !ok {
		jsonError(w, "category not found", http.StatusNotFound)
		return
	}
	type categoryDetail struct {
		Name       string    `json:"name"`
		DocCount   int       `json:"doc_count"`
		VectorDims int       `json:"vector_dims"`
		CreatedAt  time.Time `json:"created_at"`
	}
	jsonResponse(w, categoryDetail{
		Name:       c.Name,
		DocCount:   h.store.DocCountByCategory(name),
		VectorDims: len(c.Centroid),
		CreatedAt:  c.CreatedAt,
	}, http.StatusOK)
}

func (h *Handler) search(w http.ResponseWriter, r *http.Request) {
	var q search.Query
	if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if q.Q == "" {
		jsonError(w, "q is required", http.StatusBadRequest)
		return
	}

	results, err := h.searcher.Search(r.Context(), q)
	if err != nil {
		log.Printf("handler: search error: %v", err)
		jsonError(w, "search failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, results, http.StatusOK)
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, map[string]any{
		"status":         "ok",
		"doc_count":      h.store.DocCount(),
		"category_count": h.store.CategoryCount(),
	}, http.StatusOK)
}

func jsonResponse(w http.ResponseWriter, data any, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, msg string, status int) {
	jsonResponse(w, map[string]string{"error": msg}, status)
}

func queryInt(r *http.Request, key string, def int) int {
	v := r.URL.Query().Get(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}
