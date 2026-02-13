# Migration to Direct Mistral AI API

## Overview

The system has been updated to use the **direct Mistral AI API** instead of OpenRouter. This provides:
- Direct access to Mistral models
- Better control over API configuration
- Simplified authentication
- Official Mistral AI support

---

## What Changed

### **1. API Endpoint**
- **Before**: `https://openrouter.ai/api/v1`
- **After**: `https://api.mistral.ai/v1`

### **2. API Key**
- **Before**: `OPENROUTER_API_KEY`
- **After**: `MISTRAL_API_KEY`

### **3. Default Model**
- **Before**: `nvidia/nemotron-3-nano-30b-a3b:free` (via OpenRouter)
- **After**: `mistral-medium` (direct Mistral API)

### **4. HTTP Headers**
Simplified headers - removed OpenRouter-specific requirements:
- ❌ Removed: `HTTP-Referer`
- ❌ Removed: `X-Title`
- ✅ Kept: `Authorization` and `Content-Type`

---

## Files Updated

### **Backend Code**
- ✅ [backend/llm.py](backend/llm.py) - Updated API endpoint, authentication, and error messages
- ✅ [backend/main.py](backend/main.py) - Updated environment variable validation
- ✅ [backend/config.py](backend/config.py) - Already configured for Mistral (no changes needed)

### **Configuration**
- ✅ [.env](.env) - Updated `OPENROUTER_API_KEY` → `MISTRAL_API_KEY`
- ✅ [.env.example](.env.example) - Updated template

### **Documentation**
- ✅ [README.md](README.md) - Updated all references
- ✅ [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Already referenced Mistral API

---

## How to Get Your Mistral API Key

1. **Sign up** at https://console.mistral.ai
2. **Navigate** to the API Keys section
3. **Create** a new API key
4. **Copy** your key (starts with `sk-...`)
5. **Add** to your `.env` file:
   ```env
   MISTRAL_API_KEY=your_actual_mistral_api_key_here
   ```

---

## Available Mistral Models

You can configure the model in your `.env` file using `LLM_MODEL`:

| Model | Description | Use Case |
|-------|-------------|----------|
| `mistral-tiny` | Smallest, fastest | Simple queries, high volume |
| `mistral-small` | Balanced | General purpose |
| **`mistral-medium`** | **Recommended** | **Best balance of quality/speed** |
| `mistral-large-latest` | Most capable | Complex reasoning, production |
| `open-mixtral-8x7b` | Open source | Budget-friendly |

**Current default**: `mistral-medium`

---

## Configuration

### **Environment Variables**

```env
# Required
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional (defaults provided)
LLM_MODEL=mistral-medium
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1000
```

### **Updating Your Model**

Edit your `.env` file:

```env
# For faster responses (budget-friendly)
LLM_MODEL=mistral-small

# For best quality (production)
LLM_MODEL=mistral-large-latest

# For balanced performance (recommended)
LLM_MODEL=mistral-medium
```

Then restart the backend:
```bash
cd backend
python main.py
```

---

## Testing

### **Verify API Connection**

```bash
# Test API key validity
curl https://api.mistral.ai/v1/models \
  -H "Authorization: Bearer YOUR_MISTRAL_API_KEY"
```

Should return a list of available models.

### **Test Backend**

```bash
# Start backend
cd backend
python main.py

# In another terminal, test query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'
```

---

## Cost Comparison

### **Mistral AI Direct Pricing** (as of Feb 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| mistral-tiny | €0.14 | €0.42 |
| mistral-small | €0.60 | €1.80 |
| **mistral-medium** | **€2.50** | **€7.50** |
| mistral-large-latest | €8.00 | €24.00 |

**Average query cost** (mistral-medium):
- Input: ~500 tokens = €0.00125
- Output: ~200 tokens = €0.0015
- **Total: ~€0.00275 per query**

### **vs OpenRouter**
- OpenRouter pricing varies by model and markup
- Direct Mistral API typically 10-20% cheaper
- Better rate limits on direct API
- First-party support

---

## Troubleshooting

### **Error: "MISTRAL_API_KEY environment variable not set"**

**Solution**: 
1. Check your `.env` file exists in the project root
2. Verify it contains: `MISTRAL_API_KEY=sk-...`
3. Restart the backend

### **Error: "Invalid Mistral API key"**

**Solution**:
1. Verify your key at https://console.mistral.ai
2. Ensure key is copied completely (should start with `sk-`)
3. Check for extra spaces or quotes in `.env` file

### **Error: "Rate limit exceeded"**

**Solution**:
- Mistral free tier has rate limits
- Upgrade your Mistral account for higher limits
- Implement request throttling in your application

### **Error: "Model not found"**

**Solution**:
1. Check available models at https://docs.mistral.ai/models
2. Verify `LLM_MODEL` in `.env` matches available models
3. Try default: `LLM_MODEL=mistral-medium`

---

## Rollback (If Needed)

If you need to rollback to OpenRouter:

1. **Revert backend/llm.py**:
   ```python
   API_BASE = "https://openrouter.ai/api/v1"
   DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
   self.api_key = os.getenv('OPENROUTER_API_KEY')
   ```

2. **Update .env**:
   ```env
   OPENROUTER_API_KEY=your_openrouter_key
   LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
   ```

3. **Revert headers** in `llm.py`:
   ```python
   headers = {
       "Authorization": f"Bearer {self.api_key}",
       "Content-Type": "application/json",
       "HTTP-Referer": "http://localhost:3000",
       "X-Title": "Material Pricing AI Assistant"
   }
   ```

---

## Benefits of Direct Mistral API

✅ **Performance**: Direct connection, no middleware  
✅ **Cost**: Lower pricing, no OpenRouter markup  
✅ **Support**: Official Mistral AI support  
✅ **Features**: Access to latest Mistral features first  
✅ **Reliability**: First-party service uptime  
✅ **Control**: Fine-grained model configuration  

---

## Additional Resources

- **Mistral AI Documentation**: https://docs.mistral.ai
- **API Reference**: https://docs.mistral.ai/api
- **Model Comparison**: https://docs.mistral.ai/models
- **Console**: https://console.mistral.ai
- **Pricing**: https://mistral.ai/pricing

---

## Migration Complete! ✅

Your system is now configured to use the direct Mistral AI API. All references to OpenRouter have been removed and replaced with Mistral-specific configuration.

**Next Steps**:
1. Get your Mistral API key from https://console.mistral.ai
2. Update your `.env` file with `MISTRAL_API_KEY=your_key`
3. Start the backend: `cd backend && python main.py`
4. Test with a query!

---

**Questions?** Check the [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for full setup guide.
