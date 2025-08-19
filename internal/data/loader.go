package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"ml-classify-go/internal/classifier"
)

// LoadTrainingDataFromCSV loads training data from a CSV file
func LoadTrainingDataFromCSV(filename string) ([]classifier.TrainingData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []classifier.TrainingData

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
			data = append(data, classifier.TrainingData{
				Category: record[categoryIndex],
				Text:     record[textIndex],
			})
		}
	}

	return data, nil
}

// LoadTextDataFromCSV loads text data from a CSV file (text column only)
func LoadTextDataFromCSV(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var texts []string

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	// Find text column index
	var textIndex int = -1
	for i, colName := range header {
		if strings.ToLower(colName) == "text" {
			textIndex = i
			break
		}
	}

	if textIndex == -1 {
		return nil, fmt.Errorf("CSV header must contain 'text' column")
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if len(record) > textIndex {
			texts = append(texts, record[textIndex])
		}
	}

	return texts, nil
}

// ClassificationResult represents a classification result
type ClassificationResult struct {
	Text       string
	Predicted  string
	Confidence float64
}

// WriteClassificationResults writes classification results to a CSV file
func WriteClassificationResults(filename string, results []ClassificationResult) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"text", "predicted_category", "confidence"}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write results
	for _, result := range results {
		record := []string{
			result.Text,
			result.Predicted,
			fmt.Sprintf("%.4f", result.Confidence),
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
