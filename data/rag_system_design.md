# RAG System Design for an Internal Knowledge Assistant

## Background

A product team wants an assistant that can answer questions about onboarding, deployment workflows, service ownership, and troubleshooting steps. The company already has documentation spread across markdown handbooks, engineering runbooks, and support notes, but employees waste time searching across disconnected folders.

## Goal

Build a retrieval-augmented generation system that finds relevant internal documents before producing an answer. The assistant should reduce hallucinations by grounding its responses in retrieved text and should clearly separate retrieved context from generated synthesis.

## Proposed Architecture

The ingestion pipeline reads markdown and text files from trusted directories, chunks them into semantically coherent segments, and stores those segments with metadata. Each stored record includes the source path, document identifier, document type, and department. The retrieval layer embeds user questions, performs similarity search, and optionally applies metadata filters when the question is scoped to a specific team.

The application layer takes the top retrieved chunks and constructs a prompt that instructs the language model to answer only from the supplied evidence. If the retrieval results are weak or contradictory, the assistant should say so explicitly instead of pretending the answer is complete.

## Evaluation Plan

The team should measure retrieval quality with realistic employee questions such as "How do I deploy the billing API?" or "Who owns the checkout service?" Success is not only whether the answer sounds fluent, but whether the retrieved evidence is relevant, traceable, and up to date.

A useful test plan includes comparing chunking strategies, checking whether metadata filters improve relevance, and recording failure cases. Example failure cases might include outdated documents outranking current runbooks, small chunks losing critical caveats, or multilingual content confusing the embedding model.

## Operational Considerations

As the document set grows, the team should track re-indexing behavior, document deletion, and source freshness. The system should also log which chunks were retrieved so reviewers can inspect why a given answer was produced. That feedback loop is essential for improving both the data and the prompting strategy.
