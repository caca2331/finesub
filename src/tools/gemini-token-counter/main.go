package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"
	"google.golang.org/genai/tokenizer"
)

func main() {
	model := flag.String("model", "gemini-2.5-flash", "Gemini model name")
	file := flag.String("file", "", "Input text file path. If empty, read from args or stdin.")
	flag.Parse()

	var text string

	if *file != "" {
		data, err := os.ReadFile(*file)
		if err != nil {
			log.Fatalf("failed to read file: %v", err)
		}
		text = string(data)
	} else if flag.NArg() > 0 {
		text = strings.Join(flag.Args(), " ")
	} else {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			log.Fatalf("failed to read stdin: %v", err)
		}
		text = string(data)
	}

	tok, err := tokenizer.NewLocalTokenizer(*model)
	if err != nil {
		log.Fatalf("failed to init tokenizer: %v", err)
	}

	contents := []*genai.Content{
		genai.NewContentFromText(text, "user"),
	}

	resp, err := tok.CountTokens(contents, nil)
	if err != nil {
		log.Fatalf("failed to count tokens: %v", err)
	}

	fmt.Println(resp.TotalTokens)
}
