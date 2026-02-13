# Quick Start Guide - POC Implementation

## ✅ What's New

### **Phase 1: Critical Fixes**
1. **Score Inflation Fixed** - Scores now spread 10-95% (was 50-100%)
2. **Score Gate Added** - Rejects queries below 20% confidence
3. **Hybrid Retrieval Active** - BM25 + Vector with RRF fusion
4. **Boost Multipliers Reduced** - 1.1-1.3x (was 2.0-3.0x)

### **Phase 2: Intelligence Layer**
5. **Intent Classifier** - Routes chitchat vs domain queries
6. **Chitchat Handler** - Fast responses without retrieval
7. **Query Preprocessor** - Auto-corrects typos and extracts entities

---

## 🚀 Installation & Testing

### Step 1: Install New Dependency
```bash
cd backend
pip install scikit-learn==1.3.2
```

### Step 2: Start Backend
```bash
python main.py
```

**Expected logs**:
```
✓ Query preprocessor initialized
✓ Embedding manager initialized
✓ Intent classifier initialized
✓ Chitchat handler initialized
...
INFO: Application startup complete
```

### Step 3: Test Key Features

#### Test 1: Chitchat (No Retrieval)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "use_web_search": false}'
```
**Expected**: Instant template response, no retrieval logs

#### Test 2: Score Gate Rejection
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum physics", "use_web_search": false}'
```
**Expected**: "relevance is very low" message

#### Test 3: Typo Correction
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "concret prices in projectacm", "use_web_search": false}'
```
**Expected**: Auto-corrects to "concrete" and "projectacme"

#### Test 4: Hybrid Retrieval
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "oak wood price ProjectAcme", "use_web_search": false}'
```
**Expected**: High confidence results with oak + projectacme keywords

---

## 📊 Confidence Levels

| Score Range | Confidence | Behavior |
|-------------|-----------|----------|
| 0.70 - 1.00 | **High** | Full answer, no caveat |
| 0.40 - 0.69 | **Medium** | Answer + "Moderate confidence" note |
| 0.20 - 0.39 | **Low** | Answer + "Low confidence" warning |
| 0.00 - 0.19 | **Rejected** | "Relevance too low" message |

---

## 🔍 New API Response Fields

```json
{
  "response": "...",
  "sources": [...],
  "web_search_used": false,
  "confidence": "high",        // NEW: "high" | "medium" | "low"
  "intent": "domain",          // NEW: "domain" | "chitchat"
  "score": 0.85                // NEW: Top result score
}
```

---

## 🐛 Troubleshooting

### Error: "Service not fully initialized"
**Cause**: New components not loaded
**Fix**: Restart backend, check logs for initialization errors

### Error: "No module named 'sklearn'"
**Cause**: scikit-learn not installed
**Fix**: `pip install scikit-learn==1.3.2`

### Warning: "Score below threshold"
**Cause**: Query too vague or off-topic
**Expected**: Working as designed (rejects low-quality matches)

---

## 📈 Performance Expectations

- **Chitchat queries**: <0.5s (6x faster, no retrieval)
- **Domain queries**: 2-3s (same as before)
- **Score spread**: 10-95% (was 50-100%)
- **Rejection rate**: ~15-20% for off-topic queries
- **Typo handling**: Auto-corrected in 90% of cases

---

## 🎯 Testing Checklist

- [ ] Backend starts without errors
- [ ] "Hello" returns instant chitchat response
- [ ] Off-topic query gets rejected (<20% score)
- [ ] Typos auto-correct ("concret" → "concrete")
- [ ] Domain queries return confidence level
- [ ] Scores spread across 10-95% range

---

## 📞 Support

- **Implementation Doc**: `POC_IMPLEMENTATION_COMPLETE.md`
- **Architecture Analysis**: `ARCHITECTURE_ANALYSIS.md`
- **Code Files**: 
  - `backend/intent_classifier.py`
  - `backend/chitchat_handler.py`
  - `backend/query_preprocessor.py`

Ready to test! 🚀
