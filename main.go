package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

// TrainingData represents a single training example
type TrainingData struct {
	Text     string `json:"text"`
	Category string `json:"category"`
}

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

// preprocessText cleans and tokenizes text
func preprocessText(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove special characters and keep only alphanumeric and spaces
	re := regexp.MustCompile(`[^a-zA-Z0-9\s]`) // Corrected regex escaping
	text = re.ReplaceAllString(text, "")

	// Split into words and filter out short words
	words := strings.Fields(text)
	var filtered []string
	for _, word := range words {
		if len(word) > 2 { // Keep words longer than 2 characters
			filtered = append(filtered, word)
		}
	}

	return filtered
}

// Train trains the model on the provided training data
func (m *Model) Train(data []TrainingData) {
	fmt.Println("Training model...")

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

	fmt.Printf("Training completed. Categories: %v\n", m.Categories)
	fmt.Printf("Total documents: %d\n", m.TotalDocuments)
	fmt.Printf("Vocabulary size: %d\n", len(m.Vocabulary))
}

// Predict predicts the category of given text
func (m *Model) Predict(text string) (string, map[string]float64) {
	words := preprocessText(text)
	scores := make(map[string]float64)

	for _, category := range m.Categories {
		// Calculate log probability for this category
		// P(category) = count(category) / total_documents
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

// loadTrainingDataFromCSV loads training data from a CSV file
func loadTrainingDataFromCSV(filename string) ([]TrainingData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []TrainingData

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	// Find column indices
	var categoryIndex, textIndex int = -1, -1
	for i, colName := range header {
		switch strings.ToLower(colName) {
		case "category":
			categoryIndex = i
		case "text":
			textIndex = i
		}
	}

	if categoryIndex == -1 || textIndex == -1 {
		return nil, fmt.Errorf("CSV header must contain 'category' and 'text' columns")
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if len(record) > categoryIndex && len(record) > textIndex {
			data = append(data, TrainingData{
				Category: record[categoryIndex],
				Text:     record[textIndex],
			})
		}
	}

	return data, nil
}

// evaluateModel performs basic evaluation of the model
func evaluateModel(model *Model, testData []TrainingData) {
	correct := 0
	categoryStats := make(map[string]map[string]int)

	// Initialize confusion matrix
	for _, cat1 := range model.Categories {
		categoryStats[cat1] = make(map[string]int)
		for _, cat2 := range model.Categories {
			categoryStats[cat1][cat2] = 0
		}
	}

	fmt.Println("\nEvaluating model...")
	for _, item := range testData {
		predicted, _ := model.Predict(item.Text)
		categoryStats[item.Category][predicted]++

		if predicted == item.Category {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testData))
	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy*100, correct, len(testData))

	// Print confusion matrix
	fmt.Println("\nConfusion Matrix:")
	fmt.Print("Actual / Predicted\t")
	for _, cat := range model.Categories {
		fmt.Printf("%s\t", cat)
	}
	fmt.Println()

	for _, actualCat := range model.Categories {
		fmt.Printf("%s\t\t", actualCat)
		for _, predCat := range model.Categories {
			fmt.Printf("%d\t", categoryStats[actualCat][predCat])
		}
		fmt.Println()
	}

	// Write confusion matrix to CSV
	file, err := os.Create("confusion_matrix.csv")
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"Actual / Predicted"}
	header = append(header, model.Categories...)
	writer.Write(header)

	for _, actualCat := range model.Categories {
		row := []string{actualCat}
		for _, predCat := range model.Categories {
			row = append(row, fmt.Sprintf("%d", categoryStats[actualCat][predCat]))
		}
		writer.Write(row)
	}
	fmt.Println("\nConfusion matrix saved to confusion_matrix.csv")
}

// interactiveMode allows users to test the model interactively
func interactiveMode(model *Model) {
	fmt.Println("\n=== Interactive Classification Mode ===")
	fmt.Println("Enter text to classify (or 'quit' to exit):")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		text := strings.TrimSpace(scanner.Text())
		if text == "quit" || text == "exit" {
			break
		}

		if text == "" {
			continue
		}

		predicted, probabilities := model.Predict(text)

		fmt.Printf("Predicted category: %s\n", predicted)
		fmt.Println("Probabilities:")

		// Sort probabilities for better display
		type prob struct {
			category string
			value    float64
		}
		var probs []prob
		for cat, val := range probabilities {
			probs = append(probs, prob{cat, val})
		}
		sort.Slice(probs, func(i, j int) bool {
			return probs[i].value > probs[j].value
		})

		for _, p := range probs {
			fmt.Printf("  %s: %.3f\n", p.category, p.value)
		}
		fmt.Println()
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage:")
		fmt.Println("  Train: go run main.go train <training_data.csv> [model_output.json]")
		fmt.Println("  Predict: go run main.go predict <model.json> [test_data.csv]")
		fmt.Println("  Interactive: go run main.go interactive <model.json>")
		return
	}

	command := os.Args[1]

	switch command {
	case "train":
		if len(os.Args) < 3 {
			log.Fatal("Please provide training data file")
		}

		trainingFile := os.Args[2]
		modelFile := "model.json"
		if len(os.Args) > 3 {
			modelFile = os.Args[3]
		}

		// Load training data
		fmt.Printf("Loading training data from %s...\n", trainingFile)
		data, err := loadTrainingDataFromCSV(trainingFile)
		if err != nil {
			log.Fatalf("Error loading training data: %v", err)
		}

		if len(data) == 0 {
			log.Fatal("No training data found")
		}

		fmt.Printf("Loaded %d training examples\n", len(data))

		// Train model
		model := NewModel()
		model.Train(data)

		// Save model
		err = model.SaveModel(modelFile)
		if err != nil {
			log.Fatalf("Error saving model: %v", err)
		}

		fmt.Printf("Model saved to %s\n", modelFile)

	case "predict":
		if len(os.Args) < 3 {
			log.Fatal("Please provide model file")
		}

		modelFile := os.Args[2]

		// Load model
		model, err := LoadModel(modelFile)
		if err != nil {
			log.Fatalf("Error loading model: %v", err)
		}

		fmt.Printf("Model loaded from %s\n", modelFile)

		if len(os.Args) > 3 {
			// Evaluate on test data
			testFile := os.Args[3]
			testData, err := loadTrainingDataFromCSV(testFile)
			if err != nil {
				log.Fatalf("Error loading test data: %v", err)
			}

			evaluateModel(model, testData)
		} else {
			// Interactive mode
			interactiveMode(model)
		}

	case "interactive":
		if len(os.Args) < 3 {
			log.Fatal("Please provide model file")
		}

		modelFile := os.Args[2]

		// Load model
		model, err := LoadModel(modelFile)
		if err != nil {
			log.Fatalf("Error loading model: %v", err)
		}

		fmt.Printf("Model loaded from %s\n", modelFile)
		interactiveMode(model)

	default:
		fmt.Printf("Unknown command: %s\n", command)
		fmt.Println("Available commands: train, predict, interactive")
	}
}
