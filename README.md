# minijax

Interpreter-style imports:

```hs
import MiniJax.Tagless -- core language (JaxSym, foo, interpret)
import MiniJax.Tagless.Eval -- Eval interpreter
import MiniJax.Tagless.JVP.Dynamic -- untagged JVP interpreter
```

Shared core types:

```hs
import MiniJax.Common
```

Convenience entry point (core types only):

```hs
import MiniJax
```
