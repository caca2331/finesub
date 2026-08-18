package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"
	"google.golang.org/genai/tokenizer"
)

const defaultServerIdleTimeout = 300 * time.Second
const maxIdleTimeoutMilliseconds = int64((1<<63 - 1) / int64(time.Millisecond))

type countFunc func(string) (int64, error)

type serverRequest struct {
	Text          string `json:"text"`
	IdleTimeoutMS *int64 `json:"idle_timeout_ms,omitempty"`
}

type serverResponse struct {
	Tokens *int64 `json:"tokens,omitempty"`
	Error  string `json:"error,omitempty"`
}

type readEvent struct {
	line []byte
	err  error
}

func main() {
	model := flag.String("model", "gemini-2.5-flash", "Gemini model name")
	file := flag.String("file", "", "Input text file path. If empty, read from args or stdin.")
	server := flag.Bool("server", false, "Run as a persistent JSON-lines server over stdin/stdout.")
	idleTimeout := flag.Duration(
		"idle-timeout",
		defaultServerIdleTimeout,
		"Initial and minimum server idle timeout (for example 300s or 5m).",
	)
	flag.Parse()

	if *server && (*file != "" || flag.NArg() > 0) {
		log.Fatal("-server cannot be combined with -file or positional text")
	}
	if *server && *idleTimeout <= 0 {
		log.Fatal("-idle-timeout must be greater than zero")
	}

	// The experimental tokenizer currently prints its startup warning to
	// stdout. Keep server stdout protocol-clean by redirecting only the
	// single-threaded initialization window to stderr.
	protocolStdout := os.Stdout
	os.Stdout = os.Stderr
	tok, err := tokenizer.NewLocalTokenizer(*model)
	os.Stdout = protocolStdout
	if err != nil {
		log.Fatalf("failed to init tokenizer: %v", err)
	}
	count := func(text string) (int64, error) {
		contents := []*genai.Content{
			genai.NewContentFromText(text, "user"),
		}
		resp, err := tok.CountTokens(contents, nil)
		if err != nil {
			return 0, err
		}
		return int64(resp.TotalTokens), nil
	}

	if *server {
		if err := runServer(os.Stdin, os.Stdout, *idleTimeout, count); err != nil {
			log.Fatalf("server failed: %v", err)
		}
		return
	}

	text, err := readOneShotInput(*file, flag.Args(), os.Stdin)
	if err != nil {
		log.Fatalf("failed to read input: %v", err)
	}
	total, err := count(text)
	if err != nil {
		log.Fatalf("failed to count tokens: %v", err)
	}
	fmt.Println(total)
}

func readOneShotInput(file string, args []string, stdin io.Reader) (string, error) {
	if file != "" {
		data, err := os.ReadFile(file)
		if err != nil {
			return "", err
		}
		return string(data), nil
	}
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}
	data, err := io.ReadAll(stdin)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func runServer(
	input io.Reader,
	output io.Writer,
	defaultIdle time.Duration,
	count countFunc,
) error {
	if defaultIdle <= 0 {
		return errors.New("default idle timeout must be greater than zero")
	}

	events := readJSONLines(input)
	writer := bufio.NewWriter(output)
	timer := time.NewTimer(defaultIdle)
	defer timer.Stop()
	deadline := time.Now().Add(defaultIdle)

	for {
		select {
		case <-timer.C:
			return nil
		case event, ok := <-events:
			if !ok {
				return nil
			}
			if len(event.line) == 0 {
				if event.err == nil {
					continue
				}
				if errors.Is(event.err, io.EOF) {
					return nil
				}
				return fmt.Errorf("read request: %w", event.err)
			}

			remaining := time.Until(deadline)
			if remaining < 0 {
				remaining = 0
			}
			stopAndDrainTimer(timer)

			request, requestErr := decodeRequest(event.line)
			response := serverResponse{}
			if requestErr != nil {
				response.Error = requestErr.Error()
			} else {
				total, err := count(request.Text)
				if err != nil {
					response.Error = fmt.Sprintf("failed to count tokens: %v", err)
				} else {
					response.Tokens = &total
				}
			}

			if err := writeResponse(writer, response); err != nil {
				return err
			}

			nextIdle := defaultIdle
			if requestErr == nil {
				var err error
				nextIdle, err = chooseNextIdle(
					defaultIdle,
					remaining,
					request.IdleTimeoutMS,
				)
				if err != nil {
					// decodeRequest validates this value, so reaching this path
					// would indicate a programming error rather than bad input.
					return err
				}
			} else if remaining > nextIdle {
				nextIdle = remaining
			}
			deadline = time.Now().Add(nextIdle)
			timer.Reset(nextIdle)

			if event.err != nil {
				if errors.Is(event.err, io.EOF) {
					return nil
				}
				return fmt.Errorf("read request: %w", event.err)
			}
		}
	}
}

func decodeRequest(line []byte) (serverRequest, error) {
	var request serverRequest
	if err := json.Unmarshal(bytes.TrimSpace(line), &request); err != nil {
		return serverRequest{}, fmt.Errorf("invalid request JSON: %w", err)
	}
	if err := validateRequestedIdle(request.IdleTimeoutMS); err != nil {
		return serverRequest{}, err
	}
	return request, nil
}

func chooseNextIdle(
	defaultIdle time.Duration,
	previousRemaining time.Duration,
	requestedMS *int64,
) (time.Duration, error) {
	if defaultIdle <= 0 {
		return 0, errors.New("default idle timeout must be greater than zero")
	}
	if requestedMS != nil {
		if err := validateRequestedIdle(requestedMS); err != nil {
			return 0, err
		}
		return time.Duration(*requestedMS) * time.Millisecond, nil
	}
	if previousRemaining > defaultIdle {
		return previousRemaining, nil
	}
	return defaultIdle, nil
}

func validateRequestedIdle(requestedMS *int64) error {
	if requestedMS == nil {
		return nil
	}
	if *requestedMS <= 0 {
		return errors.New("idle_timeout_ms must be greater than zero")
	}
	if *requestedMS > maxIdleTimeoutMilliseconds {
		return errors.New("idle_timeout_ms is too large")
	}
	return nil
}

func readJSONLines(input io.Reader) <-chan readEvent {
	events := make(chan readEvent, 1)
	go func() {
		defer close(events)
		reader := bufio.NewReader(input)
		for {
			line, err := reader.ReadBytes('\n')
			if len(bytes.TrimSpace(line)) > 0 || err != nil {
				events <- readEvent{line: line, err: err}
			}
			if err != nil {
				return
			}
		}
	}()
	return events
}

func stopAndDrainTimer(timer *time.Timer) {
	if timer.Stop() {
		return
	}
	select {
	case <-timer.C:
	default:
	}
}

func writeResponse(writer *bufio.Writer, response serverResponse) error {
	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("encode response: %w", err)
	}
	if _, err := writer.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("write response: %w", err)
	}
	if err := writer.Flush(); err != nil {
		return fmt.Errorf("flush response: %w", err)
	}
	return nil
}
