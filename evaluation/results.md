# Hybrid RAG Evaluation Results

## Golden Evaluation Dataset

The evaluation dataset contains 55 hand-written test cases grounded in
the project's enterprise-document corpus.

Dataset composition:
- 50 answerable questions
- 5 deliberately unanswerable questions
- 34 direct questions
- 11 multi-document questions
- 5 ambiguous questions
- 5 unanswerable questions

Each evaluation case contains expected source documents and a
human-written reference answer.

## Retrieval Evaluation

Evaluation dataset:
- 50 answerable questions
- 82 expected source-document references

Results:
- Questions with at least one relevant source retrieved: 50/50
- Retrieval Hit Rate: 100.00%
- Expected sources retrieved: 66/82
- Overall Source Recall: 80.49%

Retrieval Hit Rate measures whether at least one expected relevant
source was present in the final retrieved results for each question.

Overall Source Recall measures the proportion of all expected source
documents that appeared in the final retrieved results.

The 100% hit rate therefore does not imply perfect retrieval. The
80.49% source recall shows that some multi-document questions retrieved
at least one relevant source while missing additional expected sources.

## Chunking Configuration Evaluation

Three fixed-size overlapping chunk configurations were evaluated using
the 50 answerable golden questions.

### Small
- Chunk size: 400 characters
- Chunk overlap: 75 characters
- Generated chunks: 87
- Retrieval Hit Rate: 100.00%
- Expected sources retrieved: 70/82
- Overall Source Recall: 85.37%

### Current Production Configuration
- Chunk size: 800 characters
- Chunk overlap: 150 characters
- Generated chunks: 45
- Retrieval Hit Rate: 100.00%
- Expected sources retrieved: 66/82
- Overall Source Recall: 80.49%

### Large
- Chunk size: 1200 characters
- Chunk overlap: 200 characters
- Generated chunks: 30
- Retrieval Hit Rate: 100.00%
- Expected sources retrieved: 69/82
- Overall Source Recall: 84.15%

All three configurations retrieved at least one expected source for
every answerable question.

The 400-character configuration achieved the highest overall source
recall at 85.37%. However, retrieval metrics alone do not establish
that it produces the best final generated answers because smaller
chunks may contain less surrounding context. The production
configuration therefore remains unchanged pending broader
generation-quality evaluation.

## Generation Evaluation

The generation evaluator supports:

- Semantic similarity between generated answers and human-written
  reference answers
- Answer-correctness pass-rate reporting using a documented semantic
  similarity threshold
- Results grouped by question type
- Hallucination-resistance testing on unsupported questions
- Citation provenance confidence reporting

The expanded 55-question generation evaluation has not yet produced a
complete result.

During the attempted evaluation run, the Gemini API returned a
temporary `503 UNAVAILABLE` response before the first question was
evaluated. No 0% result is reported because zero questions were
actually processed.

The generation evaluation should be rerun when the external model
service is available.

## API Tests

Automated FastAPI test suite:
- Tests executed: 7
- Tests passed: 7
- Tests failed: 0

Coverage includes:
- Healthy pipeline status
- Unavailable pipeline status
- Successful question answering
- Empty-question validation
- Missing-question validation
- Invalid question-type validation
- Unexpected pipeline failure handling

## Citation Verification

The RAG pipeline verifies returned citation metadata against the
retrieved document chunks.

Citation confidence represents citation provenance verification only.
It does not represent the probability that the generated answer is
factually correct.

A real end-to-end Docker test produced verified citations for the
retrieved enterprise documents and 100% citation provenance confidence
for the tested response.

## Evaluation Limitations

The benchmark is project-specific and uses a small synthetic enterprise
document corpus. Results should not be interpreted as general-purpose
RAG benchmark performance.

Source-level retrieval evaluation verifies whether expected documents
were retrieved; it does not independently verify that every required
passage or fact was retrieved.

Semantic similarity in the generation evaluator is an automated proxy
for reference-answer agreement and should not be interpreted as a
perfect factual-correctness metric.

Generation evaluation also depends on an external Gemini API and can
therefore be affected by temporary service availability and rate
limits.