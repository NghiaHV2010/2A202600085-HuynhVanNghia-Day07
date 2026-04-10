# Chunking Experiment Report

## Purpose

This report summarizes a small experiment comparing fixed-size chunking, sentence-based chunking, and recursive chunking on internal documentation. The objective was to understand how chunk boundaries affect retrieval quality, context preservation, and the usefulness of returned passages.

## Fixed-Size Chunking

Fixed-size chunking was simple to implement and produced predictable chunk counts. It worked reasonably well for long technical documents because every chunk stayed below a target size. However, some chunks split explanations in awkward places, especially when a procedure spanned multiple sentences. In those cases, search results sometimes returned a fragment that mentioned the right keyword but omitted the actual instruction.

## Sentence-Based Chunking

Sentence-based chunking improved readability because each chunk aligned with natural language boundaries. This made manual inspection easier and often produced more coherent retrieval results for short policy documents and FAQs. The downside was that chunk sizes became less consistent, and some dense sections still exceeded ideal embedding length when too many long sentences were grouped together.

## Recursive Chunking

Recursive chunking offered the best balance in the experiment. It first tried to split on larger structural boundaries such as paragraphs, then fell back to smaller separators only when needed. As a result, most chunks preserved context while still staying within the target size range. For the tested data, recursive chunking produced the most consistently useful passages for downstream question answering.

## Conclusion

The experiment suggests that there is no universal best strategy, but recursive chunking is a strong default for mixed technical documentation. Teams should still validate this assumption with their own queries, because retrieval quality depends on both the document style and the kinds of questions users actually ask.
