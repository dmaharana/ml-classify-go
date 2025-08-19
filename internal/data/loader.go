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