package airuntime

import (
	"strings"
	"unicode"
)

// Emission — токен CTC после схлопывания и его примерная позиция в отсчётах окна.
type Emission struct {
	Token  int
	Sample int
}

// piece преобразует строку токена в печатный текст; ▁ обозначает пробел.
func piece(token string) string {
	switch token {
	case "<unk>":
		return " ⁇ "
	case "<s>", "</s>":
		return ""
	}
	return strings.ReplaceAll(token, "▁", " ")
}

// textAndPositions собирает текст и примерные позиции начала слов (в отсчётах окна).
func textAndPositions(emissions []Emission, vocab []string) (string, []int) {
	var text strings.Builder
	positions := []int{}
	inWord := false
	for _, e := range emissions {
		value := piece(vocab[e.Token])
		text.WriteString(value)
		for _, r := range value {
			if unicode.IsSpace(r) {
				inWord = false
			} else if !inWord {
				positions = append(positions, e.Sample)
				inWord = true
			}
		}
	}
	return strings.TrimSpace(text.String()), positions
}

// normalized — только для сравнения при склейке: без регистра и пунктуации по краям.
func normalized(word string) string {
	return strings.ToLower(strings.TrimFunc(word, unicode.IsPunct))
}

// overlapSuffix ищет совпадение конца старого текста с начальной областью нового
// и возвращает новый текст без повторённых слов.
func overlapSuffix(previous, next string, limit int) string {
	a, b := strings.Fields(previous), strings.Fields(next)
	limit = min(len(a), len(b), 64, limit)
	for n := limit; n > 0; n-- {
		// Сначала пробуем самое длинное совпадение, затем более короткие.
		for offset := limit - n; offset >= 0; offset-- {
			if wordsEqual(a[len(a)-n:], b[offset:offset+n]) {
				return strings.Join(b[offset+n:], " ")
			}
		}
	}
	return next
}

func wordsEqual(a, b []string) bool {
	for i := range a {
		if normalized(a[i]) != normalized(b[i]) {
			return false
		}
	}
	return true
}

// Merger склеивает тексты окон, убирая повторно распознанные слова на перекрытии.
type Merger struct {
	previousEnd   int
	previousStart int
	previousWords []int
	lastText      string
	parts         []string
}

func NewMerger() *Merger {
	return &Merger{previousEnd: -1}
}

// Add добавляет текст окна [start, end) и возвращает часть, которая реально
// добавилась к расшифровке (без дубля перекрытия).
func (m *Merger) Add(start, end int, text string, positions []int) string {
	addition := text
	if text != "" && start < m.previousEnd {
		// Ищем дубли только около пересечения окон, а не во всём тексте.
		// Допуски 3200 (0,2 с) и 1600 (0,1 с) учитывают неточность позиции токена.
		suffix := 0
		for _, p := range m.previousWords {
			if m.previousStart+p >= start-3200 {
				suffix++
			}
		}
		if suffix < len(m.previousWords) {
			suffix++
		}
		prefix := 0
		for _, p := range positions {
			if start+p <= m.previousEnd+1600 {
				prefix++
			}
		}
		addition = overlapSuffix(m.lastText, text, min(suffix, prefix))
	}
	if addition != "" {
		m.parts = append(m.parts, addition)
	}
	m.lastText = text
	m.previousWords = positions
	m.previousStart = start
	m.previousEnd = end
	return addition
}

// Text возвращает всю склеенную расшифровку.
func (m *Merger) Text() string {
	return strings.Join(m.parts, " ")
}
