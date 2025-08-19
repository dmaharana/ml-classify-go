package classifier

import (
	"encoding/json"
	"math"
	"os"
)

// Model represents our Naive Bayes classifier
type Model struct {
	Categories     []string                  `json:"categories"`
	CategoryCounts map[string]int            `json:"category_counts"`
	WordCounts     map[string]map[string]int `json:"word_counts"`
	Vocabulary     map[string]int            `json:"vocabulary"`
	TotalWords     map[string]int            `json:"total_words"`
	TotalDocuments int                       `json:"total_documents"`
	Smoothing      float64                   `json:"smoothing"`
}

// TrainingData represents a single training example
type TrainingData struct {
	Text     string `json:"text"`
	Category string `json:"category"`
}

// NewModel creates a new Naive Bayes model
func NewModel() *Model {
	return &Model{
		Categories:     make([]string, 0),
		CategoryCounts: make(map[string]int),
		WordCounts:     make(map[string]map[string]int),
		Vocabulary:     make(map[string]int),
		TotalWords:     make(map[string]int),
		Smoothing:      1.0, // Laplace smoothing
	}
}

// Train trains the model on the provided training data
func (m *Model) Train(data []TrainingData) {
	// Reset model state to ensure training from scratch
	m.Categories = make([]string, 0)
	m.CategoryCounts = make(map[string]int)
	m.WordCounts = make(map[string]map[string]int)
	m.Vocabulary = make(map[string]int)
	m.TotalWords = make(map[string]int)
	m.TotalDocuments = 0
	// Count categories and initialize data structures
	categorySet := make(map[string]bool)
	for _, item := range data {
		if !categorySet[item.Category] {
			categorySet[item.Category] = true
			m.Categories = append(m.Categories, item.Category)
			m.CategoryCounts[item.Category] = 0
			m.WordCounts[item.Category] = make(map[string]int)
			m.TotalWords[item.Category] = 0
		}
	}

	m.TotalDocuments = len(data)

	// Process each training example
	for _, item := range data {
		m.CategoryCounts[item.Category]++
		words := preprocessText(item.Text)

		for _, word := range words {
			m.WordCounts[item.Category][word]++
			m.TotalWords[item.Category]++
			m.Vocabulary[word]++
		}
	}
}

// Predict predicts the category of given text
func (m *Model) Predict(text string) (string, map[string]float64) {
	if m.TotalDocuments == 0 || len(m.Categories) == 0 {
		return "", make(map[string]float64)
	}
	words := preprocessText(text)
	scores := make(map[string]float64)

	for _, category := range m.Categories {
		// Calculate log probability for this category
		categoryProb := math.Log(float64(m.CategoryCounts[category]) / float64(m.TotalDocuments))

		// P(word|category) for each word
		wordProb := 0.0
		for _, word := range words {
			wordCount := float64(m.WordCounts[category][word])
			totalWordsInCategory := float64(m.TotalWords[category])
			vocabularySize := float64(len(m.Vocabulary))

			// Laplace smoothing
			prob := (wordCount + m.Smoothing) / (totalWordsInCategory + vocabularySize*m.Smoothing)
			wordProb += math.Log(prob)
		}

		scores[category] = categoryProb + wordProb
	}

	// Find the category with highest score
	var bestCategory string
	bestScore := math.Inf(-1)

	for category, score := range scores {
		if score > bestScore {
			bestScore = score
			bestCategory = category
		}
	}

	// Convert log scores to probabilities for display
	probabilities := make(map[string]float64)
	maxScore := math.Inf(-1)
	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}

	total := 0.0
	for category, score := range scores {
		probabilities[category] = math.Exp(score - maxScore)
		total += probabilities[category]
	}

	// Normalize probabilities
	for category := range probabilities {
		probabilities[category] /= total
	}

	return bestCategory, probabilities
}

// SaveModel saves the model to a JSON file
func (m *Model) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(m)
}

// LoadModel loads a model from a JSON file
func LoadModel(filename string) (*Model, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var model Model
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&model)
	return &model, err
}