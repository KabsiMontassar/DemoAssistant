# Architecture Gap Analysis & POC Recommendations

## 🔴 Critical Issues Identified

### 1. **Score Inflation Problem (Always > 50%)**

**Root Causes:**
- **Aggressive boost multipliers** in `retrieval.py` (2.0x - 3.0x) inflate scores artificially
- **Normalization by max score** always makes top result = 100%, compressing score range
- **No absolute scoring reference** - everything is relative to the best match
- **Score calculation flow**:
  ```
  Base similarity (0.7) 
  → Apply 3.0x boost for project match 
  → Score becomes 2.1
  → Normalize by max (2.1) 
  → Final score = 100%
  ```

**Evidence in Code:**
```python
# retrieval.py line 166-178
PROJECT_WEIGHT = 3.0  # TOO HIGH
CATEGORY_WEIGHT = 3.0  # TOO HIGH

# Line 326-331 - Normalization crushes variance
max_score = max(doc['score'] for doc in retrieved_docs)
for doc in retrieved_docs:
    normalized_score = doc['score'] / max_score  # Always makes top = 1.0
```

### 2. **Irrelevant Results (Outside Keywords)**

**Root Causes:**
- **Hybrid retrieval NOT activated** - BM25 exists but main.py doesn't use it
- **Pure semantic search** finds conceptually similar but contextually wrong results
- **No keyword grounding** to anchor results to query terms

**Evidence:**
```python
# main.py line 301-308
retrieved = retriever_manager.retrieve(...)  # ❌ Only vector search
# hybrid_retriever exists but is NEVER CALLED

# hybrid_retrieval.py has full BM25 implementation BUT it's dormant
```

### 3. **Missing Intent Gate (No Low-Score Rejection)**

**Root Causes:**
- **No threshold gate** between retrieval and LLM
- **Binary domain check** (yes/no) instead of confidence score
- **Missing components** from architecture diagram:
  - ❌ IntentClassifier with embedding_threshold_check()
  - ❌ ChitchatHandler for direct responses
  - ❌ ScoreGate to reject results below 15-20%

**Current vs. Proposed:**
```
CURRENT:
Query → PromptVerifier (fix typos) → Retrieval → Reranking → LLM
         ↓ (binary: domain relevant?)
         
PROPOSED:
Query → PromptVerifier → IntentClassifier (confidence score)
                         ↓ (chitchat)      ↓ (domain)
                   ChitchatHandler    Retrieval → ScoreGate (15-20%)
                         ↓ (direct)            ↓ (filtered)
                         LLM                   LLM
```

---

## 📊 Architecture Comparison Matrix

| Component | Diagram Status | Implementation Status | Impact |
|-----------|----------------|----------------------|---------|
| **IntentClassifier** | ✅ Required | ❌ Missing | HIGH - No confidence scoring |
| **ChitchatHandler** | ✅ Required | ⚠️ Partial (binary check) | MEDIUM - Can't handle greetings efficiently |
| **Hybrid Retrieval (BM25+Vector)** | ✅ Required | ⚠️ Built but INACTIVE | HIGH - Pure semantic = irrelevant results |
| **RRF (Rank Fusion)** | ✅ Required | ❌ Missing | MEDIUM - No score normalization |
| **ScoreGate** | ✅ Required | ❌ Missing | HIGH - No threshold filtering |
| **CrossEncoder Reranking** | ✅ Required | ✅ Active (query similarity) | GOOD |
| **PII Redactor** | ✅ Required | ❌ Missing | LOW - Not security critical for POC |

---

## 🎯 POC Recommendations (Priority Order)

### **Priority 1: Fix Score Inflation** 🔥
**Problem**: Scores always 50%+ due to aggressive boosting + normalization

**Solution**:
1. **Reduce boost multipliers** from 2.0-3.0x to 1.1-1.3x
2. **Keep absolute scores** - don't normalize by max
3. **Use percentile-based confidence** instead of max-relative

**Implementation**:
```python
# retrieval.py - BEFORE:
PROJECT_WEIGHT = 3.0  # Too high
CATEGORY_WEIGHT = 3.0

# AFTER:
PROJECT_WEIGHT = 1.2  # Modest boost
CATEGORY_WEIGHT = 1.15
```

**Expected Impact**: Scores will spread across 10-95% range, making 15-20% threshold meaningful

---

### **Priority 2: Add Score Gate (Threshold Filter)** 🚪
**Problem**: No rejection of low-relevance results (< 15-20%)

**Solution**:
1. Add `MINIMUM_RELEVANCE_THRESHOLD = 0.20` (20%)
2. Check after reranking, before LLM
3. Return "No relevant results" if all below threshold

**Implementation**:
```python
# main.py - After reranking
MINIMUM_RELEVANCE_THRESHOLD = 0.20

filtered_results = [
    chunk for chunk in reranked 
    if chunk.get('scores', {}).get('overall_score', 0) >= MINIMUM_RELEVANCE_THRESHOLD
]

if not filtered_results:
    return "No relevant information found for this query."
```

**Expected Impact**: 
- ✅ Blocks answers when confidence < 20%
- ✅ Reduces hallucination
- ✅ Clear user communication

---

### **Priority 3: Activate Hybrid Retrieval** 🔀
**Problem**: BM25 implemented but not used; pure semantic search misses keyword matches

**Solution**:
1. Replace `retriever_manager.retrieve()` with `hybrid_retriever.retrieve()`
2. Combine vector + BM25 scores using RRF (Reciprocal Rank Fusion)
3. Weight: 60% vector, 40% keyword

**Implementation**:
```python
# main.py line 301-308
# BEFORE:
retrieved = retriever_manager.retrieve(query=fixed_query, top_k=10)

# AFTER:
# Get vector results
vector_results = retriever_manager.retrieve(query=fixed_query, top_k=20)

# Prepare for hybrid
chunks = [{"text": r["content"], "metadata": {...}} for r in vector_results]
vector_scores = [r["score"] for r in vector_results]

# Apply hybrid retrieval
retrieved = hybrid_retriever.retrieve(
    query=fixed_query,
    chunks=chunks,
    vector_scores=vector_scores,
    top_k=10
)
```

**Expected Impact**:
- ✅ Keyword grounding prevents semantic drift
- ✅ Better precision for specific queries
- ✅ Balanced semantic + lexical matching

---

### **Priority 4: Add Intent Confidence Scoring** 🎲
**Problem**: Binary domain check (yes/no) instead of confidence score (0-100%)

**Solution**:
1. Add embedding-based intent classification
2. Return confidence score instead of boolean
3. Route to chitchat handler if confidence < 30%

**Implementation**:
```python
# New file: intent_classifier.py
class IntentClassifier:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        # Domain keywords for similarity check
        self.domain_embedding = embedding_manager.model.encode(
            "material pricing construction project concrete wood metal stone supplier"
        )
    
    def classify_query(self, query: str) -> dict:
        query_embedding = self.embedding_manager.model.encode(query)
        
        # Cosine similarity to domain embedding
        similarity = cosine_similarity([query_embedding], [self.domain_embedding])[0][0]
        confidence = similarity * 100  # Convert to percentage
        
        intent = "domain" if confidence >= 30 else "chitchat"
        
        return {
            "intent": intent,
            "confidence": confidence,
            "requires_retrieval": confidence >= 30
        }
```

**Expected Impact**:
- ✅ Greetings handled directly (no retrieval)
- ✅ Confidence score visible to user
- ✅ Faster response for chitchat

---

## 🔬 POC Implementation Order

**Phase 1: Stop the Bleeding** (1-2 hours)
1. ✅ Fix score inflation (reduce boost multipliers)
2. ✅ Add score gate threshold (20%)

**Phase 2: Improve Accuracy** (2-3 hours)
3. ✅ Activate hybrid retrieval
4. ✅ Test with edge cases

**Phase 3: Polish** (1-2 hours)
5. ✅ Add intent confidence scoring
6. ✅ Add chitchat handler

**Total POC Time**: ~6 hours

---

## 📏 Success Metrics for POC

| Metric | Current (Bad) | Target (POC) |
|--------|---------------|--------------|
| **Score Range** | 50-100% | 10-95% |
| **Irrelevant Results** | ~40% | <15% |
| **Low-confidence Blocks** | 0% | ~20% of off-topic queries |
| **Response Speed** | Good | Maintain |
| **Hallucination Rate** | High | Reduced by 50% |

---

## ⚠️ What NOT to Do in POC

❌ **Don't implement full CrossEncoder** - Current reranking is sufficient  
❌ **Don't add PII Redaction** - Not security critical for POC  
❌ **Don't over-optimize** - Focus on the 4 priorities above  
❌ **Don't change embedding model** - multilingual-e5-base is good  
❌ **Don't rewrite LLM layer** - Mistral + Ollama fallback works  

---

## 🎬 Next Steps

1. **Review this analysis** - Confirm priorities align with your goals
2. **Start with Phase 1** - Fix score inflation + add gate (quick wins)
3. **Validate with test queries** - Use your worst-case examples
4. **Iterate** - Adjust thresholds based on results

**Ready to implement? Let me know which priority to start with!**
