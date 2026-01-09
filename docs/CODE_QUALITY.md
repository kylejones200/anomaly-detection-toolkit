# Code Quality Guidelines

This document outlines the code quality patterns and lessons learned to prevent CI issues and cascading errors.

## Optional Dependencies Pattern

### Use String Forward References

**✅ DO:**
```python
def _create_model(self) -> "Sequential":
    """Create model."""
    if not KERAS_AVAILABLE or Sequential is None:
        raise ImportError("Keras/TensorFlow is required.")
    return Sequential(...)
```

**❌ DON'T:**
```python
def _create_model(self) -> Sequential:  # Causes ImportError if Sequential not available
    """Create model."""
    return Sequential(...)
```

### Import Pattern

**✅ DO:**
```python
# Import only in try/except, use string forward refs in type hints
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment,misc]

def _create_model(self) -> "nn.Module":
    if not TORCH_AVAILABLE or torch is None or nn is None:
        raise ImportError("PyTorch is required.")
    return Autoencoder(...)
```

**❌ DON'T:**
```python
# Importing in TYPE_CHECKING AND try/except causes redefinition warnings
if TYPE_CHECKING:
    import torch
try:
    import torch  # F811: redefinition
```

## Defensive Checks

### Initialize Attributes

**✅ DO:**
```python
def __init__(self):
    super().__init__()
    self.model_ = None
    self.device_ = None
    self.reconstruction_errors_ = None
    # Initialize all attributes to avoid AttributeError
```

### Check Before Use

**✅ DO:**
```python
def score_samples(self, X):
    if self.model_ is None:
        raise ValueError("Detector must be fitted before scoring.")
    if not TORCH_AVAILABLE or torch is None:
        raise ImportError("PyTorch is required.")
    if self.device_ is None:
        raise RuntimeError("Device not initialized.")
    # Now safe to use
    self.model_.eval()
```

**❌ DON'T:**
```python
def score_samples(self, X):
    # No checks - can cause AttributeError or TypeError
    self.model_.eval()
    X_tensor = torch.from_numpy(X)  # torch might be None
```

### Check Optional Dependencies

**✅ DO:**
```python
if not KERAS_AVAILABLE or Sequential is None or EarlyStopping is None:
    raise ImportError("Keras/TensorFlow is required.")
```

**❌ DON'T:**
```python
# Assuming Sequential is always available
model = Sequential(...)  # Can fail if import failed
```

## Base Class Pattern

**✅ DO:**
```python
class StatisticalDetector(BaseDetector):
    def __init__(self, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.threshold: Optional[float] = None  # Initialize attribute

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Detector must be fitted before prediction.")
        scores = self.score_samples(X)
        return np.where(scores > self.threshold, -1, 1)
```

**❌ DON'T:**
```python
class StatisticalDetector(BaseDetector):
    # No __init__ - threshold not initialized

    def predict(self, X):
        scores = self.score_samples(X)
        return np.where(scores > self.threshold, -1, 1)  # AttributeError if not set
```

## Platform-Specific Considerations

### File Handling (if needed)

**✅ DO:**
```python
with open(filename, 'r') as f:
    data = f.read()
# File automatically closed
os.remove(filename)  # Safe on all platforms
```

**❌ DON'T:**
```python
f = open(filename, 'r')
data = f.read()
# Forgot to close - can cause issues on Windows
os.remove(filename)  # May fail on Windows if file still open
```

## Summary

1. **String Forward References**: Use `-> "Type"` for optional dependency types
2. **Defensive Checks**: Always check `is None` before using optional dependencies
3. **Initialize Attributes**: Set all attributes to `None` or default values in `__init__`
4. **Clear Error Messages**: Provide helpful error messages when dependencies missing
5. **Test Without Optional Deps**: Ensure core functionality works without optional packages

These patterns prevent cascading CI errors and make the codebase more robust.
