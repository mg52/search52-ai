package handler

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/mg52/search52-ai/internal/index"
)

type Handler struct {
	idx *index.Index
}

func New(idx *index.Index) *Handler {
	return &Handler{idx: idx}
}

func (h *Handler) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /documents", h.createDocument)
	mux.HandleFunc("POST /documents/batch", h.batchDocuments)
	mux.HandleFunc("DELETE /documents/{id}", h.deleteDocument)
	mux.HandleFunc("POST /search", h.search)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, map[string]string{"status": "ok"}, http.StatusOK)
	})
	return mux
}

type documentReq struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

func (d documentReq) validate() string {
	if d.ID == "" {
		return "id is required"
	}
	if d.Content == "" {
		return "content is required"
	}
	return ""
}

func (h *Handler) createDocument(w http.ResponseWriter, r *http.Request) {
	var req documentReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if msg := req.validate(); msg != "" {
		jsonError(w, msg, http.StatusBadRequest)
		return
	}
	doc, err := h.idx.Add(r.Context(), req.ID, req.Content)
	if err != nil {
		log.Printf("handler: add %s: %v", req.ID, err)
		jsonError(w, "labeling failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, doc, http.StatusCreated)
}

type batchError struct {
	ID    string `json:"id"`
	Error string `json:"error"`
}

// batchDocuments indexes many documents in one request. Individual failures
// are collected and reported without aborting the rest of the batch.
func (h *Handler) batchDocuments(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Documents []documentReq `json:"documents"`
	}
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
	for _, d := range req.Documents {
		if msg := d.validate(); msg != "" {
			errs = append(errs, batchError{ID: d.ID, Error: msg})
			continue
		}
		if _, err := h.idx.Add(r.Context(), d.ID, d.Content); err != nil {
			errs = append(errs, batchError{ID: d.ID, Error: err.Error()})
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

func (h *Handler) deleteDocument(w http.ResponseWriter, r *http.Request) {
	if !h.idx.Remove(r.PathValue("id")) {
		jsonError(w, "document not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) search(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Q string `json:"q"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Q == "" {
		jsonError(w, "q is required", http.StatusBadRequest)
		return
	}
	docs, labels, err := h.idx.Search(r.Context(), req.Q)
	if err != nil {
		log.Printf("handler: search error: %v", err)
		jsonError(w, "search failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	jsonResponse(w, map[string]any{"labels": labels, "documents": docs}, http.StatusOK)
}

func jsonResponse(w http.ResponseWriter, data any, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, msg string, status int) {
	jsonResponse(w, map[string]string{"error": msg}, status)
}
