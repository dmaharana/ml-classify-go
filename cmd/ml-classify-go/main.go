package main

import (
	"fmt"
	"log"
	"os"

	"ml-classify-go/internal/classifier"
	"ml-classify-go/internal/data"
	"ml-classify-go/internal/evaluation"
	"ml-classify-go/internal/ui"
)

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
		handleTrainCommand()
	case "predict":
		handlePredictCommand()
	case "interactive":
		handleInteractiveCommand()
	default:
		fmt.Printf("Unknown command: %s\n", command)
		fmt.Println("Available commands: train, predict, interactive")
	}
}

func handleTrainCommand() {
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
	trainingData, err := data.LoadTrainingDataFromCSV(trainingFile)
	if err != nil {
		log.Fatalf("Error loading training data: %v", err)
	}

	if len(trainingData) == 0 {
		log.Fatal("No training data found")
	}

	fmt.Printf("Loaded %d training examples\n", len(trainingData))

	// Train model
	fmt.Println("Training model...")
	model := classifier.NewModel()
	model.Train(trainingData)

	fmt.Printf("Training completed. Categories: %v\n", model.Categories)
	fmt.Printf("Total documents: %d\n", model.TotalDocuments)
	fmt.Printf("Vocabulary size: %d\n", len(model.Vocabulary))

	// Save model
	err = model.SaveModel(modelFile)
	if err != nil {
		log.Fatalf("Error saving model: %v", err)
	}

	fmt.Printf("Model saved to %s\n", modelFile)
}

func handlePredictCommand() {
	if len(os.Args) < 3 {
		log.Fatal("Please provide model file")
	}

	modelFile := os.Args[2]

	// Load model
	model, err := classifier.LoadModel(modelFile)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	fmt.Printf("Model loaded from %s\n", modelFile)

	if len(os.Args) > 3 {
		// Evaluate on test data
		testFile := os.Args[3]
		testData, err := data.LoadTrainingDataFromCSV(testFile)
		if err != nil {
			log.Fatalf("Error loading test data: %v", err)
		}

		evaluation.EvaluateModel(model, testData)
	} else {
		// Interactive mode
		ui.InteractiveMode(model)
	}
}

func handleInteractiveCommand() {
	if len(os.Args) < 3 {
		log.Fatal("Please provide model file")
	}

	modelFile := os.Args[2]

	// Load model
	model, err := classifier.LoadModel(modelFile)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	fmt.Printf("Model loaded from %s\n", modelFile)
	ui.InteractiveMode(model)
}
