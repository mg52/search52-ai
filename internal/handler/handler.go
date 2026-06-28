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
	store    store.Store
	tagger   *pipeline.Tagger
	searcher *search.Engine
}

func New(st store.Store, tagger *pipeline.Tagger, searcher *search.Engine) *Handler {
	return &Handler{store: st, tagger: tagger, searcher: searcher}
}

func (h *Handler) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /documents/llm", h.tagByLLM)
	mux.HandleFunc("POST /documents/embed", h.tagByEmbed)
	mux.HandleFunc("GET /documents/{id}", h.getDocument)
	mux.HandleFunc("PUT /documents/{id}", h.updateDocument)
	mux.HandleFunc("DELETE /documents/{id}", h.deleteDocument)
	mux.HandleFunc("GET /documents", h.listDocuments)
	mux.HandleFunc("GET /tags/{name}", h.getTag)
	mux.HandleFunc("GET /tags", h.listTags)
	mux.HandleFunc("POST /search", h.search)
	mux.HandleFunc("GET /health", h.health)
	return mux
}

type documentReq struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

func (h *Handler) parseDocumentReq(w http.ResponseWriter, r *http.Request) (store.Document, bool) {
	var req documentReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return store.Document{}, false
	}
	if req.Content == "" {
		jsonError(w, "content is required", http.StatusBadRequest)
		return store.Document{}, false
	}
	if req.ID == "" {
		req.ID = fmt.Sprintf("%d", time.Now().UnixNano())
	}
	if _, exists := h.store.GetDocument(req.ID); exists {
		jsonError(w, "document already exists", http.StatusConflict)
		return store.Document{}, false
	}
	return store.Document{ID: req.ID, Content: req.Content, CreatedAt: time.Now()}, true
}

// tagByLLM sends the document to the LLM along with all existing tag names.
// The LLM decides which existing tags apply and whether to create new ones.
// New tags trigger background embedding generation (Phase 2).
func (h *Handler) tagByLLM(w http.ResponseWriter, r *http.Request) {
	doc, ok := h.parseDocumentReq(w, r)
	if !ok {
		return
	}

	doc, err := h.tagger.TagByLLM(r.Context(), doc)
	if err != nil {
		log.Printf("handler: llm tagging %s: %v", doc.ID, err)
		jsonError(w, "tagging failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	jsonResponse(w, doc, http.StatusCreated)
}

// tagByEmbed embeds the document content and assigns existing tags by cosine
// similarity. No LLM is involved and no new tags are created.
// Requires tags to already have vectors (use /documents/llm first to populate tags).
func (h *Handler) tagByEmbed(w http.ResponseWriter, r *http.Request) {
	doc, ok := h.parseDocumentReq(w, r)
	if !ok {
		return
	}

	result, err := h.tagger.TagByEmbedding(r.Context(), doc)
	if err != nil {
		log.Printf("handler: embed tagging %s: %v", doc.ID, err)
		jsonError(w, "tagging failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	jsonResponse(w, result, http.StatusCreated)
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

// updateDocument replaces a document's content and re-tags it through the LLM,
// so its tags (and the inverted index) stay consistent with the new content.
// Old tag associations are dropped automatically when the document is re-saved.
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
	doc, err := h.tagger.TagByLLM(r.Context(), doc)
	if err != nil {
		log.Printf("handler: updating doc %s: %v", id, err)
		jsonError(w, "tagging failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	jsonResponse(w, doc, http.StatusOK)
}

func (h *Handler) deleteDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := h.store.DeleteDocument(id); !ok {
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

func (h *Handler) listTags(w http.ResponseWriter, r *http.Request) {
	tags := h.store.ListTags()
	type tagSummary struct {
		Name        string    `json:"name"`
		DocCount    int       `json:"doc_count"`
		HasVector   bool      `json:"has_vector"`
		Description string    `json:"description,omitempty"`
		CreatedAt   time.Time `json:"created_at"`
	}
	summaries := make([]tagSummary, len(tags))
	for i, t := range tags {
		summaries[i] = tagSummary{
			Name:        t.Name,
			DocCount:    h.store.DocCountByTag(t.Name),
			HasVector:   len(t.Vector) > 0,
			Description: t.Description,
			CreatedAt:   t.CreatedAt,
		}
	}
	jsonResponse(w, summaries, http.StatusOK)
}

func (h *Handler) getTag(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	tag, ok := h.store.GetTag(name)
	if !ok {
		jsonError(w, "tag not found", http.StatusNotFound)
		return
	}
	type tagDetail struct {
		Name        string    `json:"name"`
		Description string    `json:"description"`
		DocCount    int       `json:"doc_count"`
		HasVector   bool      `json:"has_vector"`
		VectorDims  int       `json:"vector_dims"`
		Examples    []string  `json:"examples"`
		CreatedAt   time.Time `json:"created_at"`
	}
	examples := tag.Examples
	if examples == nil {
		examples = []string{}
	}
	jsonResponse(w, tagDetail{
		Name:        tag.Name,
		Description: tag.Description,
		DocCount:    h.store.DocCountByTag(name),
		HasVector:   len(tag.Vector) > 0,
		VectorDims:  len(tag.Vector),
		Examples:    examples,
		CreatedAt:   tag.CreatedAt,
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
		"status":    "ok",
		"doc_count": h.store.DocCount(),
		"tag_count": h.store.TagCount(),
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
