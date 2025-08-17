# ExperienceAI v0.3.1 Release Notes

## 🔧 Bug Fixes & Improvements

### Fixed LLM Classification Method Tracking
- **Issue**: All classifications were showing `"classification_method": "heuristic"` even when LLM classification was successful
- **Fix**: Added proper `classification_method` field to LLM classification results
- **Impact**: Now properly distinguishes between LLM-based and heuristic-based classifications

### Package Structure Cleanup
- Reorganized test files into `tests/` directory
- Removed redundant and temporary files
- Streamlined `examples/` directory with only the best examples
- Updated README with cleaner, more focused documentation

## 📊 What's New in v0.3.1

### Enhanced Classification Tracking
```python
# Now properly shows when LLM classification is used:
{
  "classification_method": "llm",  # ✅ Shows 'llm' when successful
  "reasoning": "User is stating communication preference",
  "confidence": 0.9
}

# vs heuristic fallback:
{
  "classification_method": "heuristic",  # ✅ Shows 'heuristic' for pattern matching
  "reasoning": "Contains preference indicators", 
  "confidence": 0.7
}
```

### Improved Cost Efficiency
- LLM classification now properly uses the same API key and adapter as your main chatbot
- Single LLM instance handles both chat responses and learning classification
- No additional API keys or configurations needed

## 🚀 Technical Details

### Classification Method Detection
The classifier now properly tracks whether it used:
- **LLM Classification**: Intelligent analysis using your configured LLM (OpenAI, Gemini, Claude, etc.)
- **Heuristic Classification**: Pattern matching fallback when LLM is unavailable

### Unified LLM Usage
Both your chatbot responses and learning classification use the same:
- API key (OPENAI_API_KEY, GEMINI_API_KEY, etc.)
- LLM model (gpt-3.5-turbo, gemini-pro, etc.)
- Rate limits and quotas

## 🔄 Migration

### From v0.3.0 to v0.3.1
No breaking changes! This is a bug fix release that improves the existing functionality:

```python
# Your existing code works exactly the same
classifier = AutoInteractionClassifier(llm_adapter=your_llm_adapter)
classification = classifier.classify_interaction(message, response)

# But now classification.metadata['classification_method'] properly shows:
# 'llm' when using LLM classification ✅
# 'heuristic' when falling back to pattern matching ✅
```

## 📈 Performance Impact

- **No additional API costs**: Uses existing LLM adapter
- **Better learning accuracy**: LLM classification is much more nuanced than pattern matching
- **Proper attribution**: You can now see when your system is using intelligent vs basic classification

## 🧹 Package Cleanup

### Removed Files
- Multiple redundant test files
- Temporary development scripts
- Old example files
- Build artifacts

### Organized Structure
```
experience-ai/
├── experience_ai/          # Core package
├── tests/                  # All tests organized here
├── examples/               # Best examples only
├── docs/                   # Documentation
└── README.md              # Clean, focused guide
```

---

## Next Steps

1. **Install the update**: `pip install -e .`
2. **Verify LLM classification**: Check your stored interactions now show proper `classification_method`
3. **Monitor learning**: Use the analytics to see the difference between LLM and heuristic classification accuracy

This release makes ExperienceAI more transparent about how it's learning and ensures you're getting the full benefits of LLM-powered classification! 🎯
