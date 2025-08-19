package evaluation

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"ml-classify-go/internal/classifier"
)

// EvaluateModel performs basic evaluation of the model
func EvaluateModel(model *classifier.Model, testData []classifier.TrainingData) {
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
	printConfusionMatrix(model.Categories, categoryStats)

	// Write confusion matrix to CSV
	writeConfusionMatrixCSV(model.Categories, categoryStats)
}

// printConfusionMatrix prints the confusion matrix to console
func printConfusionMatrix(categories []string, categoryStats map[string]map[string]int) {
	fmt.Println("\nConfusion Matrix:")
	fmt.Print("Actual\\Predicted\t")
	for _, cat := range categories {
		fmt.Printf("%s\t", cat)
	}
	fmt.Println()

	for _, actualCat := range categories {
		fmt.Printf("%s\t\t", actualCat)
		for _, predCat := range categories {
			fmt.Printf("%d\t", categoryStats[actualCat][predCat])
		}
		fmt.Println()
	}
}

// writeConfusionMatrixCSV writes the confusion matrix to a CSV file
func writeConfusionMatrixCSV(categories []string, categoryStats map[string]map[string]int) {
	file, err := os.Create("confusion_matrix.csv")
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"Actual\\Predicted"}
	header = append(header, categories...)
	writer.Write(header)

	for _, actualCat := range categories {
		row := []string{actualCat}
		for _, predCat := range categories {
			row = append(row, fmt.Sprintf("%d", categoryStats[actualCat][predCat]))
		}
		writer.Write(row)
	}
	fmt.Println("\nConfusion matrix saved to confusion_matrix.csv")
}