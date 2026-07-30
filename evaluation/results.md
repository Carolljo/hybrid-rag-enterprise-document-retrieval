# Hybrid RAG Evaluation Results

## Retrieval Evaluation

Evaluation dataset:
- 10 answerable enterprise-policy questions
- 19 expected source-document references

Results:
- Questions with at least one relevant source retrieved: 10/10
- Retrieval Hit Rate: 100.00%
- Expected sources retrieved: 16/19
- Overall Source Recall: 84.21%

The retrieval hit rate measures whether at least one expected relevant
source was retrieved for each question. Overall source recall measures
the proportion of all expected source documents that were retrieved.

## Hallucination Resistance

Evaluation dataset:
- 3 deliberately unanswerable questions

Results:
- Correct refusals: 3/3
- Hallucination Resistance: 100.00%

For all three unsupported questions, the system responded that the
answer could not be determined from the provided documents rather than
using outside knowledge or inventing an answer.

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

## Evaluation Scope

These results were measured on the project's 13-question evaluation
dataset. The dataset is intentionally small and project-specific, so
the results should not be interpreted as general-purpose RAG benchmark
performance.