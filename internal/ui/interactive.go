package ui

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"

	"ml-classify-go/internal/classifier"
)

// InteractiveMode allows users to test the model interactively
func InteractiveMode(model *classifier.Model) {
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