package main

import (
	"bufio"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func int64Pointer(value int64) *int64 {
	return &value
}

func TestDefaultServerIdleTimeout(t *testing.T) {
	if defaultServerIdleTimeout != 300*time.Second {
		t.Fatalf("default timeout = %v, want 300s", defaultServerIdleTimeout)
	}
}

func TestChooseNextIdleExplicitValueOverridesRemainingLease(t *testing.T) {
	got, err := chooseNextIdle(
		300*time.Second,
		20*time.Minute,
		int64Pointer(45_000),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != 45*time.Second {
		t.Fatalf("next idle = %v, want 45s", got)
	}
}

func TestChooseNextIdleWithoutValueKeepsLargerRemainingLease(t *testing.T) {
	got, err := chooseNextIdle(300*time.Second, 12*time.Minute, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != 12*time.Minute {
		t.Fatalf("next idle = %v, want 12m", got)
	}
}

func TestChooseNextIdleWithoutValueRestoresDefaultFloor(t *testing.T) {
	got, err := chooseNextIdle(300*time.Second, 10*time.Second, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != 300*time.Second {
		t.Fatalf("next idle = %v, want 300s", got)
	}
}

func TestChooseNextIdleRejectsNonPositiveValue(t *testing.T) {
	if _, err := chooseNextIdle(
		300*time.Second,
		time.Minute,
		int64Pointer(0),
	); err == nil {
		t.Fatal("expected an error for zero idle_timeout_ms")
	}
}

func TestChooseNextIdleRejectsOverflow(t *testing.T) {
	if _, err := chooseNextIdle(
		300*time.Second,
		time.Minute,
		int64Pointer(maxIdleTimeoutMilliseconds+1),
	); err == nil {
		t.Fatal("expected an error for overflowing idle_timeout_ms")
	}
}

func TestRunServerHandlesMultipleRequestsInOneProcess(t *testing.T) {
	input := strings.NewReader(
		"{\"text\":\"hello\"}\n" +
			"{\"text\":\"\\u4f60\\u597d\",\"idle_timeout_ms\":600000}\n",
	)
	var output strings.Builder
	calls := 0
	count := func(text string) (int64, error) {
		calls++
		return int64(len([]rune(text))), nil
	}

	if err := runServer(input, &output, time.Hour, count); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("count calls = %d, want 2", calls)
	}

	scanner := bufio.NewScanner(strings.NewReader(output.String()))
	var responses []serverResponse
	for scanner.Scan() {
		var response serverResponse
		if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		responses = append(responses, response)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(responses) != 2 {
		t.Fatalf("responses = %d, want 2", len(responses))
	}
	if responses[0].Tokens == nil || *responses[0].Tokens != 5 {
		t.Fatalf("first response = %#v, want 5 tokens", responses[0])
	}
	if responses[1].Tokens == nil || *responses[1].Tokens != 2 {
		t.Fatalf("second response = %#v, want 2 tokens", responses[1])
	}
}

func TestRunServerReportsBadRequestAndContinues(t *testing.T) {
	input := strings.NewReader(
		"{not json}\n" +
			"{\"text\":\"ok\"}\n",
	)
	var output strings.Builder
	count := func(text string) (int64, error) {
		return int64(len(text)), nil
	}

	if err := runServer(input, &output, time.Hour, count); err != nil {
		t.Fatal(err)
	}

	scanner := bufio.NewScanner(strings.NewReader(output.String()))
	var responses []serverResponse
	for scanner.Scan() {
		var response serverResponse
		if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		responses = append(responses, response)
	}
	if len(responses) != 2 {
		t.Fatalf("responses = %d, want 2", len(responses))
	}
	if responses[0].Error == "" {
		t.Fatalf("first response = %#v, want an error", responses[0])
	}
	if responses[1].Tokens == nil || *responses[1].Tokens != 2 {
		t.Fatalf("second response = %#v, want 2 tokens", responses[1])
	}
}

// Counts the genai v1.62.0 local tokenizer gave for the same texts (tokcount
// 1.62.0-0): the embedded vocabulary must keep them bit for bit.
func TestCounterMatchesGenaiLocalTokenizer(t *testing.T) {
	count, err := newCounter()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		want int64
	}{
		{"", 0},
		{"hello world", 2},
		{"你好，世界\nline2 🙂", 7},
		{"今日はいい天気ですね。字幕の校正をお願いします。", 12},
		{"   \n\t  ", 4},
		{"1\t00:00:01.000\t00:00:02.500\tそうですね、えーと\n2\t00:00:03.000\t00:00:04.000\tはい", 64},
		{strings.Repeat("The quick brown fox jumps over the lazy dog. ", 50), 501},
		{"안녕하세요 👋🏽 ‘quotes’ — ça va? Ελληνικά", 13},
	}
	for _, c := range cases {
		got, err := count(c.text)
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("count(%q) = %d, want %d", c.text, got, c.want)
		}
	}
}
